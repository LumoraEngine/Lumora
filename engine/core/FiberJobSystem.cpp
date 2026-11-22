#include "core/FiberJobSystem.hpp"

#include "core/Assert.hpp"

#include <algorithm>
#include <bit>
#include <exception>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <ucontext.h>
#endif

namespace lumora {

namespace detail {

namespace {

inline void cpuPause() noexcept {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#elif defined(_M_ARM64)
    __yield();
#else
    std::this_thread::yield();
#endif
}

#ifdef _WIN32
void pinThreadToCore(unsigned workerIndex) noexcept {
    DWORD_PTR processMask = 0;
    DWORD_PTR systemMask = 0;
    if (::GetProcessAffinityMask(::GetCurrentProcess(), &processMask, &systemMask) != 0 && processMask != 0) {
        const auto activeCount = static_cast<unsigned>(std::popcount(static_cast<std::uint64_t>(processMask)));
        unsigned targetOrdinal = workerIndex % std::max(1u, activeCount);
        DWORD_PTR chosenMask = 0;
        for (unsigned bit = 0; bit < static_cast<unsigned>(sizeof(DWORD_PTR) * 8); ++bit) {
            const DWORD_PTR bitMask = DWORD_PTR{1} << bit;
            if ((processMask & bitMask) != 0) {
                if (targetOrdinal == 0) {
                    chosenMask = bitMask;
                    break;
                }
                --targetOrdinal;
            }
        }
        if (chosenMask != 0) {
            ::SetThreadAffinityMask(::GetCurrentThread(), chosenMask);
        }
        return;
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned maxBits = static_cast<unsigned>(sizeof(DWORD_PTR) * 8);
    const unsigned bit = workerIndex % std::min(hw, maxBits);
    ::SetThreadAffinityMask(::GetCurrentThread(), DWORD_PTR{1} << bit);
}
#else
void pinThreadToCore(unsigned workerIndex) noexcept {
    cpu_set_t currentMask;
    CPU_ZERO(&currentMask);
    if (::pthread_getaffinity_np(::pthread_self(), sizeof(cpu_set_t), &currentMask) == 0) {
        std::vector<int> allowed;
        allowed.reserve(CPU_SETSIZE);
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &currentMask)) {
                allowed.push_back(cpu);
            }
        }
        if (!allowed.empty()) {
            cpu_set_t targetMask;
            CPU_ZERO(&targetMask);
            CPU_SET(allowed[workerIndex % allowed.size()], &targetMask);
            ::pthread_setaffinity_np(::pthread_self(), sizeof(cpu_set_t), &targetMask);
            return;
        }
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    cpu_set_t fallbackMask;
    CPU_ZERO(&fallbackMask);
    CPU_SET(static_cast<int>(workerIndex % hw), &fallbackMask);
    ::pthread_setaffinity_np(::pthread_self(), sizeof(cpu_set_t), &fallbackMask);
}
#endif

} // namespace

struct Fiber {
    FiberJobSystem* owner{nullptr};
    std::size_t index{0};

    std::move_only_function<void()> task;
    JobCounter* completionCounter{nullptr};

    Fiber* waitNext{nullptr};
    std::uint32_t waitTarget{0};

#ifdef _WIN32
    void* nativeHandle{nullptr};
#else
    static constexpr std::size_t kStackBytes = 256 * 1024;
    ucontext_t context{};
    std::unique_ptr<std::byte[]> stack;
#endif
};

struct Worker {
    enum class SwitchReason : std::uint8_t {
        None,
        Finished,
        WaitForCounter,
        Yield,
    };

    FiberJobSystem* owner{nullptr};
    unsigned index{0};
    std::thread thread;

#ifdef _WIN32
    void* schedulerFiber{nullptr};
#else
    ucontext_t schedulerContext{};
#endif

    Fiber* currentFiber{nullptr};
    SwitchReason switchReason{SwitchReason::None};
    JobCounter* waitCounter{nullptr};
    std::uint32_t waitTarget{0};
};

namespace {
thread_local Worker* t_currentWorker = nullptr;
} // namespace

#ifdef _WIN32
namespace {
VOID WINAPI win32FiberEntry(LPVOID param) {
    auto* fiber = static_cast<Fiber*>(param);
    fiber->owner->fiberEntryPoint(fiber);
}
} // namespace
#else
extern "C" {
static void lumoraFiberTrampoline() {
    Worker* worker = t_currentWorker;
    LUMORA_VERIFY(worker != nullptr && worker->currentFiber != nullptr,
                  "Fiber trampoline entered without an active worker fiber");
    Fiber* self = worker->currentFiber;
    self->owner->fiberEntryPoint(self);
}
}
#endif

namespace {

void initFiberContext(Fiber& fiber) {
#ifdef _WIN32
    fiber.nativeHandle = ::CreateFiber(0, &win32FiberEntry, &fiber);
    LUMORA_VERIFY(fiber.nativeHandle != nullptr, "CreateFiber failed");
#else
    const int rc = ::getcontext(&fiber.context);
    LUMORA_VERIFY(rc == 0, "getcontext failed");
    fiber.stack = std::make_unique<std::byte[]>(Fiber::kStackBytes);
    fiber.context.uc_stack.ss_sp = fiber.stack.get();
    fiber.context.uc_stack.ss_size = Fiber::kStackBytes;
    fiber.context.uc_link = nullptr;
    ::makecontext(&fiber.context, &lumoraFiberTrampoline, 0);
#endif
}

void destroyFiberContext(Fiber& fiber) noexcept {
#ifdef _WIN32
    if (fiber.nativeHandle != nullptr) {
        ::DeleteFiber(fiber.nativeHandle);
        fiber.nativeHandle = nullptr;
    }
#else
    fiber.stack.reset();
#endif
}

inline void switchFromSchedulerToFiber(Worker& worker, Fiber& fiber) noexcept {
#ifdef _WIN32
    (void)worker;
    ::SwitchToFiber(fiber.nativeHandle);
#else
    ::swapcontext(&worker.schedulerContext, &fiber.context);
#endif
}

inline void switchFromFiberToScheduler(Worker& worker, Fiber& fiber) noexcept {
#ifdef _WIN32
    (void)fiber;
    ::SwitchToFiber(worker.schedulerFiber);
#else
    ::swapcontext(&fiber.context, &worker.schedulerContext);
#endif
}

} // namespace

} // namespace detail

// ---------------------------------------------------------------------------
// JobCounter
// ---------------------------------------------------------------------------

JobCounter::JobCounter(std::uint32_t initialCount) noexcept
    : count_(initialCount) {}

JobCounter::~JobCounter() {
    unsigned spins = 0;
    while (inFlight_.load(std::memory_order_acquire) != 0) {
        if (++spins < 64) {
            detail::cpuPause();
        } else {
            std::this_thread::yield();
        }
    }
}

void JobCounter::increment(std::uint32_t amount) noexcept {
    if (amount == 0) {
        return;
    }
    count_.fetch_add(amount, std::memory_order_acq_rel);
}

void JobCounter::decrement(std::uint32_t amount) {
    if (amount == 0) {
        return;
    }

    inFlight_.fetch_add(1, std::memory_order_seq_cst);

    std::uint32_t prev = count_.load(std::memory_order_relaxed);
    std::uint32_t newVal = 0;
    bool updated = false;
    while (prev > 0) {
        const std::uint32_t step = std::min(prev, amount);
        newVal = prev - step;
        if (count_.compare_exchange_weak(prev, newVal, std::memory_order_seq_cst, std::memory_order_relaxed)) {
            updated = true;
            break;
        }
    }

    if (!updated) {
        inFlight_.fetch_sub(1, std::memory_order_release);
        return;
    }

    detail::Fiber* toWake = nullptr;
    {
        acquireWaitLock();
        detail::Fiber** link = &waitHead_;
        while (*link != nullptr) {
            detail::Fiber* node = *link;
            if (newVal <= node->waitTarget) {
                *link = node->waitNext;
                node->waitNext = toWake;
                toWake = node;
            } else {
                link = &node->waitNext;
            }
        }
        releaseWaitLock();
    }

    while (toWake != nullptr) {
        detail::Fiber* next = toWake->waitNext;
        toWake->waitNext = nullptr;
        toWake->owner->wakeFiber(toWake);
        toWake = next;
    }

    if (externalWaiters_.load(std::memory_order_seq_cst) > 0) {
        count_.notify_all();
    }

    inFlight_.fetch_sub(1, std::memory_order_release);
}

std::uint32_t JobCounter::value() const noexcept {
    return count_.load(std::memory_order_acquire);
}

void JobCounter::reset(std::uint32_t newValue) noexcept {
    count_.store(newValue, std::memory_order_release);
}

bool JobCounter::tryAddWaiter(detail::Fiber* fiber, std::uint32_t targetValue) noexcept {
    acquireWaitLock();
    if (count_.load(std::memory_order_acquire) <= targetValue) {
        releaseWaitLock();
        return false;
    }
    fiber->waitTarget = targetValue;
    fiber->waitNext = waitHead_;
    waitHead_ = fiber;
    releaseWaitLock();
    return true;
}

void JobCounter::waitExternal(std::uint32_t targetValue) noexcept {
    externalWaiters_.fetch_add(1, std::memory_order_seq_cst);
    for (;;) {
        const std::uint32_t current = count_.load(std::memory_order_seq_cst);
        if (current <= targetValue) {
            break;
        }
        count_.wait(current, std::memory_order_acquire);
    }
    externalWaiters_.fetch_sub(1, std::memory_order_release);
}

void JobCounter::acquireWaitLock() noexcept {
    while (waitLock_.test_and_set(std::memory_order_acquire)) {
        while (waitLock_.test(std::memory_order_relaxed)) {
            detail::cpuPause();
        }
    }
}

void JobCounter::releaseWaitLock() noexcept {
    waitLock_.clear(std::memory_order_release);
}

// ---------------------------------------------------------------------------
// FiberJobSystem
// ---------------------------------------------------------------------------

namespace {

std::size_t normalizeWorkerCount(unsigned requested) noexcept {
    if (requested != 0) {
        return requested;
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

std::size_t computeFiberQueueCapacity(std::size_t poolSize) noexcept {
    const std::size_t clamped = std::max<std::size_t>(poolSize, 1);
    return std::max<std::size_t>(std::bit_ceil(clamped * 4), 1024);
}

} // namespace

FiberJobSystem::FiberJobSystem(unsigned workerThreadCount,
                               std::size_t initialFiberPoolSize,
                               std::size_t initialJobQueueCapacity)
    : jobQueue_(std::max<std::size_t>(initialJobQueueCapacity, 64))
    , readyFibers_(computeFiberQueueCapacity(initialFiberPoolSize))
    , yieldedFibers_(computeFiberQueueCapacity(initialFiberPoolSize))
    , freeFibers_(computeFiberQueueCapacity(initialFiberPoolSize)) {
    const std::size_t initialFibers = std::max<std::size_t>(initialFiberPoolSize, 1);
    {
        std::scoped_lock lock(fiberAllocMutex_);
        allFibers_.reserve(initialFibers);
        for (std::size_t i = 0; i < initialFibers; ++i) {
            detail::Fiber* fiber = allocateFiberLocked();
            const bool pushed = freeFibers_.tryPush(fiber);
            LUMORA_VERIFY(pushed, "Failed to populate initial fiber pool");
        }
    }

    const std::size_t numWorkers = normalizeWorkerCount(workerThreadCount);
    workers_.reserve(numWorkers);
    for (std::size_t i = 0; i < numWorkers; ++i) {
        auto worker = std::make_unique<detail::Worker>();
        worker->owner = this;
        worker->index = static_cast<unsigned>(i);
        workers_.push_back(std::move(worker));
    }

    for (auto& worker : workers_) {
        detail::Worker* rawWorker = worker.get();
        rawWorker->thread = std::thread([this, rawWorker] { workerMain(*rawWorker); });
    }
}

FiberJobSystem::~FiberJobSystem() {
    shutdown();
}

void FiberJobSystem::shutdown() {
    if (shutdownStarted_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    {
        std::scoped_lock lock(wakeMutex_);
        shuttingDown_.store(true, std::memory_order_seq_cst);
        wakeCv_.notify_all();
    }

    for (auto& worker : workers_) {
        if (worker && worker->thread.joinable()) {
            worker->thread.join();
        }
    }

    {
        std::scoped_lock lock(fiberAllocMutex_);
        for (auto& fiber : allFibers_) {
            if (fiber) {
                fiber->task = {};
                detail::destroyFiberContext(*fiber);
            }
        }
        allFibers_.clear();
        allFibersCount_.store(0, std::memory_order_release);
    }

    detail::JobItem leftoverJob;
    while (jobQueue_.tryPop(leftoverJob)) {
    }
    detail::Fiber* leftoverFiber = nullptr;
    while (readyFibers_.tryPop(leftoverFiber)) {
    }
    while (yieldedFibers_.tryPop(leftoverFiber)) {
    }
    while (freeFibers_.tryPop(leftoverFiber)) {
    }

    workers_.clear();
}

void FiberJobSystem::submit(std::move_only_function<void()> job, JobCounter* counter) {
    if (!job || shuttingDown_.load(std::memory_order_acquire)) {
        return;
    }

    if (counter != nullptr) {
        counter->increment(1);
    }

    detail::JobItem item{std::move(job), counter};
    for (;;) {
        if (jobQueue_.tryPush(std::move(item))) {
            wakeWorkerIfSleeping();
            return;
        }
        if (shuttingDown_.load(std::memory_order_acquire)) {
            if (counter != nullptr) {
                counter->decrement(1);
            }
            return;
        }
        wakeWorkerIfSleeping();
        if (detail::t_currentWorker != nullptr) {
            yield();
        } else {
            std::this_thread::yield();
        }
    }
}

bool FiberJobSystem::trySubmit(std::move_only_function<void()> job, JobCounter* counter) {
    if (!job || shuttingDown_.load(std::memory_order_acquire)) {
        return false;
    }

    if (counter != nullptr) {
        counter->increment(1);
    }

    detail::JobItem item{std::move(job), counter};
    if (jobQueue_.tryPush(std::move(item))) {
        wakeWorkerIfSleeping();
        return true;
    }

    if (counter != nullptr) {
        counter->decrement(1);
    }
    return false;
}

void FiberJobSystem::waitForCounter(JobCounter& counter, std::uint32_t targetValue) {
    if (counter.value() <= targetValue) {
        return;
    }

    if (detail::Worker* worker = detail::t_currentWorker; worker != nullptr && worker->currentFiber != nullptr) {
        worker->switchReason = detail::Worker::SwitchReason::WaitForCounter;
        worker->waitCounter = &counter;
        worker->waitTarget = targetValue;
        detail::switchFromFiberToScheduler(*worker, *worker->currentFiber);
        return;
    }

    counter.waitExternal(targetValue);
}

void FiberJobSystem::yield() {
    if (detail::Worker* worker = detail::t_currentWorker; worker != nullptr && worker->currentFiber != nullptr) {
        worker->switchReason = detail::Worker::SwitchReason::Yield;
        detail::switchFromFiberToScheduler(*worker, *worker->currentFiber);
        return;
    }
    std::this_thread::yield();
}

void FiberJobSystem::parallelFor(std::size_t count, std::size_t grain,
                                 const std::function<void(std::size_t, std::size_t)>& fn) {
    if (count == 0 || !fn) {
        return;
    }

    grain = std::max<std::size_t>(grain, 1);
    const std::size_t chunks = (count + grain - 1) / grain;
    if (chunks == 1 || shuttingDown_.load(std::memory_order_acquire)) {
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t begin = c * grain;
            const std::size_t end = std::min(count, begin + grain);
            fn(begin, end);
        }
        return;
    }

    JobCounter counter;
    std::exception_ptr failure;
    std::mutex failureMutex;

    auto runRange = [&](std::size_t begin, std::size_t end) {
        try {
            fn(begin, end);
        } catch (...) {
            std::scoped_lock lock(failureMutex);
            if (!failure) {
                failure = std::current_exception();
            }
        }
    };

    struct WaitGuard {
        FiberJobSystem& sys;
        JobCounter& ctr;
        ~WaitGuard() { sys.waitForCounter(ctr, 0); }
    };

    constexpr std::size_t kDirectChunkLimit = 512;
    if (chunks <= kDirectChunkLimit) {
        WaitGuard waitGuard{*this, counter};
        for (std::size_t c = 1; c < chunks; ++c) {
            const std::size_t begin = c * grain;
            const std::size_t end = std::min(count, begin + grain);
            submit([&runRange, begin, end] { runRange(begin, end); }, &counter);
        }
        runRange(0, std::min(count, grain));
    } else {
        const std::size_t taskCount = std::min<std::size_t>(
            chunks, std::max<std::size_t>(static_cast<std::size_t>(workerCount()) * 8, 256));
        std::atomic<std::size_t> nextChunk{0};

        auto drain = [&] {
            for (std::size_t c = nextChunk.fetch_add(1, std::memory_order_relaxed);
                 c < chunks;
                 c = nextChunk.fetch_add(1, std::memory_order_relaxed)) {
                const std::size_t begin = c * grain;
                const std::size_t end = std::min(count, begin + grain);
                runRange(begin, end);
            }
        };

        WaitGuard waitGuard{*this, counter};
        for (std::size_t t = 1; t < taskCount; ++t) {
            submit([&drain] { drain(); }, &counter);
        }
        drain();
    }

    if (failure) {
        std::rethrow_exception(failure);
    }
}

bool FiberJobSystem::isOnWorkerFiber() noexcept {
    return detail::t_currentWorker != nullptr && detail::t_currentWorker->currentFiber != nullptr;
}

void FiberJobSystem::fiberEntryPoint(detail::Fiber* self) noexcept {
    for (;;) {
        if (self->task) {
            auto task = std::move(self->task);
            JobCounter* counter = self->completionCounter;
            self->completionCounter = nullptr;

            try {
                task();
            } catch (...) {
                // Fire-and-forget job threw; counter still decrements below.
            }

            // Destroy captured functors before waking any waiter on `counter`.
            task = {};

            if (counter != nullptr) {
                counter->decrement(1);
            }
        }

        detail::Worker* worker = detail::t_currentWorker;
        LUMORA_VERIFY(worker != nullptr, "Worker TLS missing on fiber return");
        worker->switchReason = detail::Worker::SwitchReason::Finished;
        detail::switchFromFiberToScheduler(*worker, *self);
    }
}

void FiberJobSystem::workerMain(detail::Worker& worker) {
    detail::pinThreadToCore(worker.index);
    detail::t_currentWorker = &worker;

#ifdef _WIN32
    worker.schedulerFiber = ::ConvertThreadToFiber(nullptr);
    LUMORA_VERIFY(worker.schedulerFiber != nullptr, "ConvertThreadToFiber failed");
#endif

    while (!shouldExit()) {
        detail::Fiber* fiber = nullptr;
        if (readyFibers_.tryPop(fiber)) {
            runFiber(worker, fiber);
            continue;
        }

        if (!jobQueue_.emptyApprox()) {
            detail::Fiber* freeFiber = acquireFiber();
            if (freeFiber != nullptr) {
                detail::JobItem job;
                if (jobQueue_.tryPop(job)) {
                    freeFiber->task = std::move(job.task);
                    freeFiber->completionCounter = job.counter;
                    runFiber(worker, freeFiber);
                    continue;
                }
                const bool returned = freeFibers_.tryPush(freeFiber);
                LUMORA_VERIFY(returned, "Failed to return unused fiber to pool");
            }
        }

        if (yieldedFibers_.tryPop(fiber)) {
            runFiber(worker, fiber);
            continue;
        }

        waitForWork();
    }

#ifdef _WIN32
    ::ConvertFiberToThread();
    worker.schedulerFiber = nullptr;
#endif
    detail::t_currentWorker = nullptr;
}

void FiberJobSystem::runFiber(detail::Worker& worker, detail::Fiber* fiber) {
    runningWorkers_.fetch_add(1, std::memory_order_acq_rel);

    worker.currentFiber = fiber;
    worker.switchReason = detail::Worker::SwitchReason::None;
    detail::switchFromSchedulerToFiber(worker, *fiber);
    worker.currentFiber = nullptr;

    switch (worker.switchReason) {
    case detail::Worker::SwitchReason::Finished: {
        worker.switchReason = detail::Worker::SwitchReason::None;
        const bool pushed = freeFibers_.tryPush(fiber);
        LUMORA_VERIFY(pushed, "freeFibers_ queue capacity exceeded");
        if (!jobQueue_.emptyApprox()) {
            wakeWorkerIfSleeping();
        }
        break;
    }

    case detail::Worker::SwitchReason::WaitForCounter: {
        JobCounter* counter = worker.waitCounter;
        const std::uint32_t target = worker.waitTarget;
        worker.waitCounter = nullptr;
        worker.waitTarget = 0;
        worker.switchReason = detail::Worker::SwitchReason::None;

        if (!counter->tryAddWaiter(fiber, target)) {
            wakeFiber(fiber);
        }
        break;
    }

    case detail::Worker::SwitchReason::Yield: {
        worker.switchReason = detail::Worker::SwitchReason::None;
        const bool pushed = yieldedFibers_.tryPush(fiber);
        LUMORA_VERIFY(pushed, "yieldedFibers_ queue capacity exceeded");
        wakeWorkerIfSleeping();
        break;
    }

    case detail::Worker::SwitchReason::None:
        break;
    }

    if (runningWorkers_.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
        shuttingDown_.load(std::memory_order_seq_cst)) {
        std::scoped_lock lock(wakeMutex_);
        wakeCv_.notify_all();
    }
}

void FiberJobSystem::waitForWork() {
    for (unsigned spin = 0; spin < 32; ++spin) {
        if (!readyFibers_.emptyApprox() ||
            !yieldedFibers_.emptyApprox() ||
            (!jobQueue_.emptyApprox() && (!freeFibers_.emptyApprox() || canGrowFiberPool())) ||
            shouldExit()) {
            return;
        }
        detail::cpuPause();
    }

    std::unique_lock lock(wakeMutex_);
    sleepingWorkers_.fetch_add(1, std::memory_order_seq_cst);

    const auto hasRunnableWork = [this]() noexcept {
        if (!readyFibers_.emptyApprox() || !yieldedFibers_.emptyApprox()) {
            return true;
        }
        if (!jobQueue_.emptyApprox() && (!freeFibers_.emptyApprox() || canGrowFiberPool())) {
            return true;
        }
        return false;
    };

    while (!hasRunnableWork() && !shouldExit()) {
        wakeCv_.wait(lock);
    }

    sleepingWorkers_.fetch_sub(1, std::memory_order_relaxed);
}

void FiberJobSystem::wakeFiber(detail::Fiber* fiber) {
    const bool pushed = readyFibers_.tryPush(fiber);
    LUMORA_VERIFY(pushed, "readyFibers_ queue capacity exceeded");
    wakeWorkerIfSleeping();
}

void FiberJobSystem::wakeWorkerIfSleeping() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (sleepingWorkers_.load(std::memory_order_relaxed) > 0) {
        std::scoped_lock lock(wakeMutex_);
        wakeCv_.notify_one();
    }
}

detail::Fiber* FiberJobSystem::acquireFiber() {
    detail::Fiber* fiber = nullptr;
    if (freeFibers_.tryPop(fiber)) {
        return fiber;
    }

    std::scoped_lock lock(fiberAllocMutex_);
    if (freeFibers_.tryPop(fiber)) {
        return fiber;
    }
    if (allFibers_.size() < freeFibers_.capacity()) {
        return allocateFiberLocked();
    }
    return nullptr;
}

detail::Fiber* FiberJobSystem::allocateFiberLocked() {
    auto fiber = std::make_unique<detail::Fiber>();
    fiber->owner = this;
    fiber->index = allFibers_.size();
    detail::initFiberContext(*fiber);
    detail::Fiber* raw = fiber.get();
    allFibers_.push_back(std::move(fiber));
    allFibersCount_.store(allFibers_.size(), std::memory_order_release);
    return raw;
}

bool FiberJobSystem::canGrowFiberPool() const noexcept {
    return allFibersCount_.load(std::memory_order_acquire) < freeFibers_.capacity();
}

bool FiberJobSystem::shouldExit() const noexcept {
    return shuttingDown_.load(std::memory_order_seq_cst) &&
           readyFibers_.emptyApprox() &&
           yieldedFibers_.emptyApprox() &&
           jobQueue_.emptyApprox() &&
           runningWorkers_.load(std::memory_order_seq_cst) == 0;
}

} // namespace lumora
