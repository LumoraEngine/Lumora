#pragma once

#include "core/LockFreeQueue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lumora {

class FiberJobSystem;
class JobCounter;

namespace detail {

struct Fiber;
struct Worker;

struct JobItem {
    std::move_only_function<void()> task;
    JobCounter* counter{nullptr};
};

} // namespace detail

/// Atomic job synchronization primitive with a lock-free / spinlock-protected wait list
/// of suspended fibers plus safe blocking support for non-worker external threads.
class JobCounter {
public:
    explicit JobCounter(std::uint32_t initialCount = 0) noexcept;
    ~JobCounter();

    JobCounter(const JobCounter&) = delete;
    JobCounter& operator=(const JobCounter&) = delete;
    JobCounter(JobCounter&&) = delete;
    JobCounter& operator=(JobCounter&&) = delete;

    void increment(std::uint32_t amount = 1) noexcept;
    void decrement(std::uint32_t amount = 1);

    [[nodiscard]] std::uint32_t value() const noexcept;
    [[nodiscard]] bool isZero() const noexcept { return value() == 0; }

    /// Resets the counter value. Only valid when no jobs or waiters are active on it.
    void reset(std::uint32_t newValue = 0) noexcept;

private:
    friend class FiberJobSystem;

    bool tryAddWaiter(detail::Fiber* fiber, std::uint32_t targetValue) noexcept;
    void waitExternal(std::uint32_t targetValue) noexcept;
    void acquireWaitLock() noexcept;
    void releaseWaitLock() noexcept;

    std::atomic<std::uint32_t> count_{0};
    std::atomic<std::uint32_t> inFlight_{0};
    std::atomic<std::uint32_t> externalWaiters_{0};
    std::atomic_flag waitLock_{};
    detail::Fiber* waitHead_{nullptr};
};

/// Naughty Dog-style cooperative fiber-based job system.
///
/// - Spawns `workerCount` OS threads pinned to CPU cores.
/// - Executes jobs on pooled fibers (`CreateFiber`/`SwitchToFiber` on Windows, `ucontext` on POSIX).
/// - `waitForCounter()` from a worker fiber suspends the fiber onto `JobCounter` and switches
///   the OS worker back to the scheduler to run other work without ever blocking the OS thread.
class FiberJobSystem {
public:
    static constexpr std::size_t kDefaultFiberPoolSize = 128;
    static constexpr std::size_t kDefaultJobQueueCapacity = 16384;

    explicit FiberJobSystem(unsigned workerThreadCount = 0,
                            std::size_t initialFiberPoolSize = kDefaultFiberPoolSize,
                            std::size_t initialJobQueueCapacity = kDefaultJobQueueCapacity);
    ~FiberJobSystem();

    FiberJobSystem(const FiberJobSystem&) = delete;
    FiberJobSystem& operator=(const FiberJobSystem&) = delete;
    FiberJobSystem(FiberJobSystem&&) = delete;
    FiberJobSystem& operator=(FiberJobSystem&&) = delete;

    /// Drains pending work and joins all worker threads cleanly. Idempotent.
    void shutdown();

    /// Enqueues a job into the lock-free MPMC queue, optionally tracking completion via `counter`.
    void submit(std::move_only_function<void()> job, JobCounter* counter = nullptr);

    /// Non-blocking submit into the lock-free MPMC queue. Returns false if shutting down or full.
    [[nodiscard]] bool trySubmit(std::move_only_function<void()> job, JobCounter* counter = nullptr);

    /// Suspends the current fiber (or blocks an external thread) until `counter.value() <= targetValue`.
    void waitForCounter(JobCounter& counter, std::uint32_t targetValue = 0);

    /// Cooperatively yields the current fiber back to the ready queue (or yields OS thread externally).
    void yield();

    /// Executes `fn(begin, end)` across `[0, count)` in chunks of at most `grain` items
    /// using fibers + `JobCounter`. Safe for arbitrary nesting even on 1 worker thread.
    void parallelFor(std::size_t count, std::size_t grain,
                     const std::function<void(std::size_t, std::size_t)>& fn);

    [[nodiscard]] unsigned workerCount() const noexcept { return static_cast<unsigned>(workers_.size()); }
    [[nodiscard]] std::size_t fiberPoolSize() const noexcept { return allFibersCount_.load(std::memory_order_acquire); }
    [[nodiscard]] std::size_t pendingJobs() const noexcept { return jobQueue_.sizeApprox(); }
    [[nodiscard]] bool isRunning() const noexcept { return !shuttingDown_.load(std::memory_order_acquire); }
    [[nodiscard]] static bool isOnWorkerFiber() noexcept;

    /// Internal entry invoked by platform fiber trampolines.
    void fiberEntryPoint(detail::Fiber* self) noexcept;

private:
    friend class JobCounter;

    void workerMain(detail::Worker& worker);
    void runFiber(detail::Worker& worker, detail::Fiber* fiber);
    void waitForWork();
    void wakeFiber(detail::Fiber* fiber);
    void wakeWorkerIfSleeping();

    [[nodiscard]] detail::Fiber* acquireFiber();
    [[nodiscard]] detail::Fiber* allocateFiberLocked();
    [[nodiscard]] bool canGrowFiberPool() const noexcept;
    [[nodiscard]] bool shouldExit() const noexcept;

    MpmcBoundedQueue<detail::JobItem> jobQueue_;
    MpmcBoundedQueue<detail::Fiber*> readyFibers_;
    MpmcBoundedQueue<detail::Fiber*> yieldedFibers_;
    MpmcBoundedQueue<detail::Fiber*> freeFibers_;

    std::mutex fiberAllocMutex_;
    std::vector<std::unique_ptr<detail::Fiber>> allFibers_;
    std::atomic<std::size_t> allFibersCount_{0};

    std::vector<std::unique_ptr<detail::Worker>> workers_;

    std::atomic<unsigned> runningWorkers_{0};
    std::atomic<unsigned> sleepingWorkers_{0};
    std::atomic<bool> shuttingDown_{false};
    std::atomic<bool> shutdownStarted_{false};

    std::mutex wakeMutex_;
    std::condition_variable wakeCv_;
};

} // namespace lumora
