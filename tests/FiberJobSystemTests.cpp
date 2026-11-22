#include "core/FiberJobSystem.hpp"
#include "core/LockFreeQueue.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace lumora;

namespace {

struct LifetimeTracker {
    std::atomic<int>* counter{nullptr};

    explicit LifetimeTracker(std::atomic<int>* c = nullptr) : counter(c) {
        if (counter) {
            counter->fetch_add(1, std::memory_order_relaxed);
        }
    }
    LifetimeTracker(LifetimeTracker&& other) noexcept : counter(other.counter) { other.counter = nullptr; }
    LifetimeTracker& operator=(LifetimeTracker&& other) noexcept {
        if (this != &other) {
            if (counter) {
                counter->fetch_sub(1, std::memory_order_relaxed);
            }
            counter = other.counter;
            other.counter = nullptr;
        }
        return *this;
    }
    ~LifetimeTracker() {
        if (counter) {
            counter->fetch_sub(1, std::memory_order_relaxed);
        }
    }
};

} // namespace

// ---------------------------------------------------------------------------
// SpscRingBuffer tests
// ---------------------------------------------------------------------------

TEST(LockFreeQueue, SpscBasicCapacityAndMoveOnlyPreservation) {
    SpscRingBuffer<std::unique_ptr<int>> q(5); // rounds up to power-of-two 8
    EXPECT_EQ(q.capacity(), 8u);
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.sizeApprox(), 0u);

    for (int i = 0; i < 8; ++i) {
        auto ptr = std::make_unique<int>(i * 10);
        EXPECT_TRUE(q.tryPush(std::move(ptr)));
        EXPECT_EQ(ptr, nullptr);
    }
    EXPECT_EQ(q.sizeApprox(), 8u);
    EXPECT_FALSE(q.empty());

    // Failed push on a full queue must NOT move from the caller's object.
    auto overflow = std::make_unique<int>(999);
    EXPECT_FALSE(q.tryPush(std::move(overflow)));
    ASSERT_NE(overflow, nullptr);
    EXPECT_EQ(*overflow, 999);

    for (int i = 0; i < 8; ++i) {
        std::unique_ptr<int> popped;
        ASSERT_TRUE(q.tryPop(popped));
        ASSERT_NE(popped, nullptr);
        EXPECT_EQ(*popped, i * 10);
    }
    EXPECT_TRUE(q.empty());
    std::unique_ptr<int> emptyPop;
    EXPECT_FALSE(q.tryPop(emptyPop));
}

TEST(LockFreeQueue, SpscDestroysRemainingElements) {
    std::atomic<int> alive{0};
    {
        SpscRingBuffer<LifetimeTracker, 16> q;
        for (int i = 0; i < 10; ++i) {
            EXPECT_TRUE(q.tryEmplace(&alive));
        }
        EXPECT_EQ(alive.load(), 10);
        for (int i = 0; i < 3; ++i) {
            LifetimeTracker tmp;
            EXPECT_TRUE(q.tryPop(tmp));
        }
        EXPECT_EQ(alive.load(), 7);
    }
    EXPECT_EQ(alive.load(), 0);
}

TEST(LockFreeQueue, Spsc100kOrderAndChecksum) {
    constexpr std::uint64_t kItemCount = 100'000;
    SpscRingBuffer<std::uint64_t> queue(256); // Small ring forces heavy wraparound & contention.

    std::uint64_t expectedChecksum = 0;
    for (std::uint64_t i = 0; i < kItemCount; ++i) {
        expectedChecksum += i * 2654435761ULL + 17ULL;
    }

    std::uint64_t actualChecksum = 0;
    bool orderValid = true;

    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kItemCount; ++i) {
            while (!queue.tryPush(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&] {
        std::uint64_t nextExpected = 0;
        while (nextExpected < kItemCount) {
            std::uint64_t value = 0;
            if (queue.tryPop(value)) {
                if (value != nextExpected) {
                    orderValid = false;
                }
                actualChecksum += value * 2654435761ULL + 17ULL;
                ++nextExpected;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_TRUE(orderValid);
    EXPECT_EQ(actualChecksum, expectedChecksum);
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.sizeApprox(), 0u);
}

// ---------------------------------------------------------------------------
// MpmcBoundedQueue tests
// ---------------------------------------------------------------------------

TEST(LockFreeQueue, MpmcBasicAndCompileTimeCapacity) {
    MpmcBoundedQueue<std::unique_ptr<int>, 4> q;
    EXPECT_EQ(q.capacity(), 4u);

    for (int i = 0; i < 4; ++i) {
        auto v = std::make_unique<int>(i + 1);
        EXPECT_TRUE(q.tryPush(std::move(v)));
    }
    auto rejected = std::make_unique<int>(42);
    EXPECT_FALSE(q.tryPush(std::move(rejected)));
    ASSERT_NE(rejected, nullptr);
    EXPECT_EQ(*rejected, 42);

    for (int i = 0; i < 4; ++i) {
        std::unique_ptr<int> out;
        ASSERT_TRUE(q.tryPop(out));
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(*out, i + 1);
    }
    std::unique_ptr<int> emptyOut;
    EXPECT_FALSE(q.tryPop(emptyOut));
}

TEST(LockFreeQueue, MpmcDestroysRemainingElements) {
    std::atomic<int> alive{0};
    {
        MpmcBoundedQueue<LifetimeTracker> q(16);
        for (int i = 0; i < 9; ++i) {
            EXPECT_TRUE(q.tryEmplace(&alive));
        }
        EXPECT_EQ(alive.load(), 9);
        for (int i = 0; i < 4; ++i) {
            LifetimeTracker g;
            EXPECT_TRUE(q.tryPop(g));
        }
        EXPECT_EQ(alive.load(), 5);
    }
    EXPECT_EQ(alive.load(), 0);
}

TEST(LockFreeQueue, Mpmc4Producers4ConsumersStress) {
    constexpr int kProducers = 4;
    constexpr int kConsumers = 4;
    constexpr int kItemsPerProducer = 25'000;
    constexpr int kTotalItems = kProducers * kItemsPerProducer;

    MpmcBoundedQueue<int> queue(512);
    std::vector<std::atomic<int>> seen(kTotalItems);
    std::atomic<int> totalConsumed{0};
    std::atomic<bool> start{false};

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    producers.reserve(kProducers);
    consumers.reserve(kConsumers);

    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const int base = p * kItemsPerProducer;
            for (int i = 0; i < kItemsPerProducer; ++i) {
                const int id = base + i;
                while (!queue.tryPush(id)) {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (int c = 0; c < kConsumers; ++c) {
        consumers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (;;) {
                int id = -1;
                if (queue.tryPop(id)) {
                    seen[static_cast<std::size_t>(id)].fetch_add(1, std::memory_order_relaxed);
                    totalConsumed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    if (totalConsumed.load(std::memory_order_acquire) >= kTotalItems) {
                        break;
                    }
                    std::this_thread::yield();
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    EXPECT_EQ(totalConsumed.load(), kTotalItems);
    EXPECT_TRUE(queue.emptyApprox());
    for (int i = 0; i < kTotalItems; ++i) {
        ASSERT_EQ(seen[static_cast<std::size_t>(i)].load(std::memory_order_relaxed), 1) << "Item id " << i;
    }
}

// ---------------------------------------------------------------------------
// FiberJobSystem tests
// ---------------------------------------------------------------------------

TEST(FiberJobSystem, DefaultConstructorUsesHardwareConcurrency) {
    FiberJobSystem jobs(0);
    EXPECT_EQ(jobs.workerCount(), std::max(1u, std::thread::hardware_concurrency()));
    EXPECT_TRUE(jobs.isRunning());
    EXPECT_FALSE(FiberJobSystem::isOnWorkerFiber());
}

TEST(FiberJobSystem, SingleWorkerFiberYieldPreventsDeadlock) {
    // CRITICAL PROOF: Only 1 OS worker thread!
    // An outer job waits on 8 inner child jobs via waitForCounter().
    // Without genuine cooperative fiber suspension/resumption, this would deadlock immediately.
    FiberJobSystem jobs(1);
    ASSERT_EQ(jobs.workerCount(), 1u);

    std::atomic<std::uint32_t> childMask{0};
    std::atomic<int> childSum{0};
    std::atomic<bool> outerFinishedAfterChildren{false};

    const auto startTime = std::chrono::steady_clock::now();

    JobCounter outerCounter;
    jobs.submit([&] {
        EXPECT_TRUE(FiberJobSystem::isOnWorkerFiber());

        JobCounter innerCounter;
        for (int i = 0; i < 8; ++i) {
            jobs.submit([&, i] {
                EXPECT_TRUE(FiberJobSystem::isOnWorkerFiber());
                childMask.fetch_or(1u << static_cast<unsigned>(i), std::memory_order_relaxed);
                childSum.fetch_add(i + 1, std::memory_order_relaxed);
            }, &innerCounter);
        }

        jobs.waitForCounter(innerCounter, 0);

        EXPECT_EQ(innerCounter.value(), 0u);
        EXPECT_EQ(childMask.load(std::memory_order_acquire), 0xFFu);
        EXPECT_EQ(childSum.load(std::memory_order_acquire), 36);
        outerFinishedAfterChildren.store(true, std::memory_order_release);
    }, &outerCounter);

    jobs.waitForCounter(outerCounter, 0);
    const auto elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();

    EXPECT_TRUE(outerFinishedAfterChildren.load());
    EXPECT_EQ(childMask.load(), 0xFFu);
    EXPECT_EQ(childSum.load(), 36);
    EXPECT_LT(elapsedMs, 5000);
}

TEST(FiberJobSystem, MultiLevelNestedFiberWaitingOnSingleWorker) {
    FiberJobSystem jobs(1);
    std::atomic<int> leafCount{0};

    JobCounter rootCounter;
    jobs.submit([&] {
        JobCounter level1;
        for (int a = 0; a < 4; ++a) {
            jobs.submit([&] {
                JobCounter level2;
                for (int b = 0; b < 4; ++b) {
                    jobs.submit([&] {
                        JobCounter level3;
                        for (int c = 0; c < 4; ++c) {
                            jobs.submit([&] { leafCount.fetch_add(1, std::memory_order_relaxed); }, &level3);
                        }
                        jobs.waitForCounter(level3, 0);
                    }, &level2);
                }
                jobs.waitForCounter(level2, 0);
            }, &level1);
        }
        jobs.waitForCounter(level1, 0);
    }, &rootCounter);

    jobs.waitForCounter(rootCounter, 0);
    EXPECT_EQ(leafCount.load(), 64);
}

TEST(FiberJobSystem, WaitForNonZeroTargetValue) {
    FiberJobSystem jobs(2);
    JobCounter counter;
    std::atomic<int> completed{0};
    std::atomic<bool> passedMidpoint{false};

    JobCounter driver;
    jobs.submit([&] {
        for (int i = 0; i < 6; ++i) {
            jobs.submit([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                completed.fetch_add(1, std::memory_order_release);
            }, &counter);
        }
        // Wait until at most 2 remain (at least 4 finished).
        jobs.waitForCounter(counter, 2);
        EXPECT_LE(counter.value(), 2u);
        EXPECT_GE(completed.load(std::memory_order_acquire), 4);
        passedMidpoint.store(true, std::memory_order_release);

        jobs.waitForCounter(counter, 0);
        EXPECT_EQ(completed.load(std::memory_order_acquire), 6);
    }, &driver);

    jobs.waitForCounter(driver, 0);
    EXPECT_TRUE(passedMidpoint.load());
}

TEST(FiberJobSystem, NestedMultiLevelParallelForSum) {
    for (unsigned workers : {1u, 4u}) {
        FiberJobSystem jobs(workers);

        constexpr std::size_t kRows = 64;
        constexpr std::size_t kCols = 128;
        std::atomic<std::uint64_t> totalSum{0};

        jobs.parallelFor(kRows, 4, [&](std::size_t rBegin, std::size_t rEnd) {
            for (std::size_t r = rBegin; r < rEnd; ++r) {
                std::atomic<std::uint64_t> rowSum{0};
                jobs.parallelFor(kCols, 16, [&](std::size_t cBegin, std::size_t cEnd) {
                    std::uint64_t localSum = 0;
                    for (std::size_t c = cBegin; c < cEnd; ++c) {
                        localSum += static_cast<std::uint64_t>(r * kCols + c);
                    }
                    rowSum.fetch_add(localSum, std::memory_order_relaxed);
                });
                totalSum.fetch_add(rowSum.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        });

        constexpr std::uint64_t kN = kRows * kCols;
        constexpr std::uint64_t kExpected = (kN * (kN - 1)) / 2;
        EXPECT_EQ(totalSum.load(), kExpected) << "workers=" << workers;
    }
}

TEST(FiberJobSystem, LargeParallelForCoversEveryIndexOnce) {
    FiberJobSystem jobs(4);
    constexpr std::size_t kCount = 50'003;
    std::vector<std::atomic<int>> hits(kCount);

    jobs.parallelFor(kCount, 17, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
            hits[i].fetch_add(1, std::memory_order_relaxed);
        }
    });

    for (std::size_t i = 0; i < kCount; ++i) {
        ASSERT_EQ(hits[i].load(std::memory_order_relaxed), 1) << "index=" << i;
    }
}

TEST(FiberJobSystem, ParallelForRethrowsAndStillCompletesAllChunks) {
    FiberJobSystem jobs(2);
    std::atomic<int> processed{0};

    EXPECT_THROW(
        jobs.parallelFor(100, 1,
                         [&](std::size_t b, std::size_t) {
                             processed.fetch_add(1, std::memory_order_relaxed);
                             if (b == 42) {
                                 throw std::logic_error("chunk 42");
                             }
                         }),
        std::logic_error);

    EXPECT_EQ(processed.load(), 100);
}

TEST(FiberJobSystem, FiberPoolGrowsWhenSuspendedFibersExceedInitialPool) {
    // Small initial pool of 16 fibers, while 48 outer fibers suspend simultaneously.
    FiberJobSystem jobs(2, 16);
    EXPECT_EQ(jobs.fiberPoolSize(), 16u);

    std::atomic<int> completed{0};
    JobCounter allOuter;

    for (int i = 0; i < 48; ++i) {
        jobs.submit([&] {
            JobCounter inner;
            for (int j = 0; j < 2; ++j) {
                jobs.submit([&] { completed.fetch_add(1, std::memory_order_relaxed); }, &inner);
            }
            jobs.waitForCounter(inner, 0);
        }, &allOuter);
    }

    jobs.waitForCounter(allOuter, 0);
    EXPECT_EQ(completed.load(), 96);
    EXPECT_GE(jobs.fiberPoolSize(), 16u);
}

TEST(FiberJobSystem, DestructorAndShutdownDrainPendingWorkCleanly) {
    std::atomic<int> executed{0};
    {
        FiberJobSystem jobs(4);
        for (int i = 0; i < 1000; ++i) {
            jobs.submit([&] { executed.fetch_add(1, std::memory_order_relaxed); });
        }
        jobs.shutdown();
        jobs.shutdown(); // Idempotent second call must be safe.
        EXPECT_FALSE(jobs.isRunning());
    }
    EXPECT_EQ(executed.load(), 1000);
}

TEST(FiberJobSystem, SingleWorkerQueueFullDoesNotDeadlock) {
    // Tiny job queue (64) on 1 worker thread: outer job submits 300 inner jobs
    // forcing submit() to cooperatively yield when the bounded queue fills up.
    FiberJobSystem jobs(1, 32, 64);
    std::atomic<int> count{0};

    JobCounter outer;
    jobs.submit([&] {
        JobCounter inner;
        for (int i = 0; i < 300; ++i) {
            jobs.submit([&] { count.fetch_add(1, std::memory_order_relaxed); }, &inner);
        }
        jobs.waitForCounter(inner, 0);
    }, &outer);

    jobs.waitForCounter(outer, 0);
    EXPECT_EQ(count.load(), 300);
}
