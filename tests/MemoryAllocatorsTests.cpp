#include "core/MemoryAllocators.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace lumora;

namespace {

[[nodiscard]] bool isAligned(const void* ptr, std::size_t align) noexcept {
    return ptr != nullptr && (reinterpret_cast<std::uintptr_t>(ptr) % align) == 0;
}

struct LiveRegion {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    void* ptr = nullptr;
    std::size_t size = 0;
    std::uint8_t pattern = 0;
};

void verifyNoOverlapsAndPatterns(const std::vector<LiveRegion>& regions) {
    for (const auto& r : regions) {
        const auto* bytes = static_cast<const std::uint8_t*>(r.ptr);
        for (std::size_t i = 0; i < r.size; ++i) {
            ASSERT_EQ(bytes[i], r.pattern) << "Memory corruption at byte " << i;
        }
    }

    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> intervals;
    intervals.reserve(regions.size());
    for (const auto& r : regions) {
        intervals.emplace_back(r.begin, r.end);
    }
    std::sort(intervals.begin(), intervals.end());
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        ASSERT_LE(intervals[i - 1].second, intervals[i].first)
            << "Overlapping ranges detected between intervals " << (i - 1) << " and " << i;
    }
}

} // namespace

// -----------------------------------------------------------------------------
// LinearAllocator Tests
// -----------------------------------------------------------------------------

TEST(LinearAllocator, DefaultCapacityAndInitialState) {
    LinearAllocator alloc;
    EXPECT_EQ(alloc.capacity(), LinearAllocator::kDefaultCapacity);
    EXPECT_EQ(alloc.used(), 0u);
    EXPECT_EQ(alloc.peak(), 0u);
    EXPECT_EQ(alloc.remaining(), alloc.capacity());
    EXPECT_TRUE(isAligned(alloc.data(), 64));
}

TEST(LinearAllocator, Enforces16_64_128ByteAlignments) {
    LinearAllocator alloc(4096);

    void* p1 = alloc.allocate(3, 16);
    ASSERT_NE(p1, nullptr);
    EXPECT_TRUE(isAligned(p1, 16));
    EXPECT_TRUE(alloc.owns(p1));

    void* p2 = alloc.allocate(5, 64);
    ASSERT_NE(p2, nullptr);
    EXPECT_TRUE(isAligned(p2, 64));
    EXPECT_TRUE(alloc.owns(p2));

    void* p3 = alloc.allocate(17, 128);
    ASSERT_NE(p3, nullptr);
    EXPECT_TRUE(isAligned(p3, 128));
    EXPECT_TRUE(alloc.owns(p3));

    void* p4 = alloc.allocate(1, 16);
    ASSERT_NE(p4, nullptr);
    EXPECT_TRUE(isAligned(p4, 16));

    EXPECT_GE(static_cast<std::byte*>(p2) - static_cast<std::byte*>(p1), 3);
    EXPECT_GE(static_cast<std::byte*>(p3) - static_cast<std::byte*>(p2), 5);
    EXPECT_GE(static_cast<std::byte*>(p4) - static_cast<std::byte*>(p3), 17);
    EXPECT_EQ(alloc.used() + alloc.remaining(), alloc.capacity());
    EXPECT_EQ(alloc.peak(), alloc.used());
}

TEST(LinearAllocator, AllocateArrayRespectsTypeAndExplicitAlignment) {
    struct alignas(64) CacheLineVec {
        float data[16];
    };
    struct alignas(128) SimdBlock {
        std::uint64_t words[16];
    };

    LinearAllocator alloc(8192);
    auto* floats = alloc.allocateArray<float>(13);
    ASSERT_NE(floats, nullptr);
    EXPECT_TRUE(isAligned(floats, alignof(float)));

    auto* cacheVecs = alloc.allocateArray<CacheLineVec>(4);
    ASSERT_NE(cacheVecs, nullptr);
    EXPECT_TRUE(isAligned(cacheVecs, 64));

    auto* simdBlocks = alloc.allocateArray<SimdBlock>(2);
    ASSERT_NE(simdBlocks, nullptr);
    EXPECT_TRUE(isAligned(simdBlocks, 128));

    auto* customAligned = alloc.allocateArray<std::uint32_t>(8, 128);
    ASSERT_NE(customAligned, nullptr);
    EXPECT_TRUE(isAligned(customAligned, 128));

    EXPECT_EQ(alloc.allocateArray<int>(0), nullptr);
    EXPECT_EQ(alloc.allocateArray<int>(100000), nullptr);
}

TEST(LinearAllocator, RewindAndResetPreservePeak) {
    LinearAllocator alloc(1024);

    void* first = alloc.allocate(128, 16);
    ASSERT_NE(first, nullptr);
    const auto m1 = alloc.marker();

    void* second = alloc.allocate(256, 64);
    ASSERT_NE(second, nullptr);
    const std::size_t peakAfterSecond = alloc.used();
    EXPECT_EQ(alloc.peak(), peakAfterSecond);

    alloc.rewind(m1);
    EXPECT_EQ(alloc.used(), m1);
    EXPECT_EQ(alloc.peak(), peakAfterSecond);
    EXPECT_EQ(alloc.remaining(), alloc.capacity() - m1);

    void* reused = alloc.allocate(64, 16);
    ASSERT_NE(reused, nullptr);
    EXPECT_EQ(reused, second); // Immediately after m1, 16B aligned
    EXPECT_EQ(alloc.peak(), peakAfterSecond);

    // Invalid forward rewind is ignored.
    alloc.rewind(alloc.capacity());
    EXPECT_LE(alloc.used(), peakAfterSecond);

    alloc.reset();
    EXPECT_EQ(alloc.used(), 0u);
    EXPECT_EQ(alloc.remaining(), alloc.capacity());
    EXPECT_EQ(alloc.peak(), peakAfterSecond);
}

TEST(LinearAllocator, FrameScopeRewindsOnDestruction) {
    LinearAllocator alloc(2048);

    void* persistent = alloc.allocate(64, 16);
    ASSERT_NE(persistent, nullptr);
    const std::size_t baseUsed = alloc.used();

    {
        LinearAllocator::FrameScope outer(alloc);
        EXPECT_EQ(outer.marker(), baseUsed);

        void* a = outer.allocate(128, 64);
        ASSERT_NE(a, nullptr);
        EXPECT_TRUE(isAligned(a, 64));
        const std::size_t afterOuterAlloc = alloc.used();
        EXPECT_GT(afterOuterAlloc, baseUsed);

        {
            FrameScope inner(alloc);
            void* b = alloc.allocate(256, 128);
            ASSERT_NE(b, nullptr);
            EXPECT_TRUE(isAligned(b, 128));
            EXPECT_GT(alloc.used(), afterOuterAlloc);
        }

        EXPECT_EQ(alloc.used(), afterOuterAlloc);
    }

    EXPECT_EQ(alloc.used(), baseUsed);
    EXPECT_GT(alloc.peak(), baseUsed);
}

TEST(LinearAllocator, OverflowAndInvalidArgumentsReturnNullptr) {
    LinearAllocator alloc(256);

    EXPECT_EQ(alloc.allocate(0, 16), nullptr);
    EXPECT_EQ(alloc.allocate(16, 0), nullptr);
    EXPECT_EQ(alloc.allocate(16, 3), nullptr);  // non-power-of-two
    EXPECT_EQ(alloc.allocate(16, 24), nullptr); // non-power-of-two
    EXPECT_EQ(alloc.allocate(257, 1), nullptr);
    EXPECT_EQ(alloc.used(), 0u);

    void* fit = alloc.allocate(200, 1);
    ASSERT_NE(fit, nullptr);
    EXPECT_EQ(alloc.used(), 200u);
    EXPECT_EQ(alloc.remaining(), 56u);

    // Alignment push + size overflows remaining 56 bytes.
    EXPECT_EQ(alloc.allocate(48, 64), nullptr);
    EXPECT_EQ(alloc.used(), 200u);

    // Exact remaining fit succeeds.
    void* tail = alloc.allocate(56, 1);
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(alloc.remaining(), 0u);
    EXPECT_EQ(alloc.allocate(1, 1), nullptr);

    LinearAllocator empty(0);
    EXPECT_EQ(empty.capacity(), 0u);
    EXPECT_EQ(empty.allocate(1, 1), nullptr);
    EXPECT_FALSE(empty.owns(fit));
}

TEST(LinearAllocator, MoveTransfersOwnership) {
    LinearAllocator a(512);
    void* ptr = a.allocate(64, 64);
    ASSERT_NE(ptr, nullptr);

    LinearAllocator b(std::move(a));
    EXPECT_EQ(a.capacity(), 0u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_TRUE(b.owns(ptr));
    EXPECT_EQ(b.used(), 64u);

    LinearAllocator c(128);
    c = std::move(b);
    EXPECT_TRUE(c.owns(ptr));
    EXPECT_EQ(b.capacity(), 0u);
}

// -----------------------------------------------------------------------------
// PoolAllocator & TypedPool<T> Tests
// -----------------------------------------------------------------------------

TEST(PoolAllocator, AlignmentAndOwnershipChecks) {
    for (std::size_t align : {16u, 64u, 128u}) {
        PoolAllocator pool(24, 8, align);
        EXPECT_EQ(pool.capacity(), 8u);
        EXPECT_EQ(pool.activeCount(), 0u);
        EXPECT_GE(pool.slotSize(), 24u);
        EXPECT_EQ(pool.slotSize() % align, 0u);

        std::vector<void*> ptrs;
        for (int i = 0; i < 8; ++i) {
            void* p = pool.allocate();
            ASSERT_NE(p, nullptr);
            EXPECT_TRUE(isAligned(p, align));
            EXPECT_TRUE(pool.owns(p));
            EXPECT_FALSE(pool.owns(static_cast<std::byte*>(p) + 1));
            ptrs.push_back(p);
        }
        EXPECT_EQ(pool.activeCount(), 8u);
        EXPECT_EQ(pool.allocate(), nullptr); // exhausted

        int stackVar = 0;
        EXPECT_FALSE(pool.owns(nullptr));
        EXPECT_FALSE(pool.owns(&stackVar));

        for (void* p : ptrs) {
            pool.deallocate(p);
        }
        EXPECT_EQ(pool.activeCount(), 0u);
    }
}

TEST(PoolAllocator, ReuseAfterDeallocateAndReset) {
    PoolAllocator pool(32, 4, 16);
    void* a = pool.allocate();
    void* b = pool.allocate();
    void* c = pool.allocate();
    void* d = pool.allocate();
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(pool.allocate(), nullptr);

    pool.deallocate(b);
    pool.deallocate(d);
    EXPECT_EQ(pool.activeCount(), 2u);

    void* r1 = pool.allocate();
    void* r2 = pool.allocate();
    EXPECT_EQ(r1, d);
    EXPECT_EQ(r2, b);
    EXPECT_EQ(pool.activeCount(), 4u);
    EXPECT_EQ(pool.allocate(), nullptr);

    pool.deallocate(nullptr); // safe no-op
    EXPECT_EQ(pool.activeCount(), 4u);

    pool.reset();
    EXPECT_EQ(pool.activeCount(), 0u);
    EXPECT_EQ(pool.availableCount(), 4u);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NE(pool.allocate(), nullptr);
    }
    EXPECT_EQ(pool.allocate(), nullptr);
}

TEST(TypedPool, InvokesConstructorsAndDestructors) {
    struct LifetimeTracker {
        int id = 0;
        int* liveCount = nullptr;
        int* dtorCount = nullptr;

        LifetimeTracker(int idVal, int* live, int* dtor)
            : id(idVal), liveCount(live), dtorCount(dtor) {
            ++(*liveCount);
        }

        ~LifetimeTracker() {
            --(*liveCount);
            ++(*dtorCount);
        }
    };

    int live = 0;
    int dtors = 0;

    {
        TypedPool<LifetimeTracker> pool(4);
        auto* o1 = pool.construct(10, &live, &dtors);
        auto* o2 = pool.construct(20, &live, &dtors);
        auto* o3 = pool.construct(30, &live, &dtors);
        auto* o4 = pool.construct(40, &live, &dtors);
        ASSERT_NE(o1, nullptr);
        ASSERT_NE(o2, nullptr);
        ASSERT_NE(o3, nullptr);
        ASSERT_NE(o4, nullptr);
        EXPECT_EQ(pool.construct(50, &live, &dtors), nullptr);
        EXPECT_EQ(live, 4);
        EXPECT_EQ(dtors, 0);
        EXPECT_EQ(o3->id, 30);

        pool.destroy(o2);
        pool.destroy(o2); // double destroy is safely ignored
        EXPECT_EQ(live, 3);
        EXPECT_EQ(dtors, 1);
        EXPECT_EQ(pool.activeCount(), 3u);

        auto* o5 = pool.construct(50, &live, &dtors);
        ASSERT_NE(o5, nullptr);
        EXPECT_EQ(o5->id, 50);
        EXPECT_EQ(live, 4);

        pool.reset();
        EXPECT_EQ(live, 0);
        EXPECT_EQ(dtors, 5);
        EXPECT_EQ(pool.activeCount(), 0u);

        (void)pool.construct(60, &live, &dtors);
        (void)pool.construct(70, &live, &dtors);
        EXPECT_EQ(live, 2);
    }

    // Pool destruction destroys any remaining live non-trivial objects.
    EXPECT_EQ(live, 0);
    EXPECT_EQ(dtors, 7);
}

TEST(TypedPool, ExceptionDuringConstructionDoesNotLeakSlot) {
    struct ThrowingObj {
        explicit ThrowingObj(bool shouldThrow) {
            if (shouldThrow) {
                throw std::runtime_error("ctor failure");
            }
        }
    };

    TypedPool<ThrowingObj> pool(2);
    EXPECT_THROW((void)pool.construct(true), std::runtime_error);
    EXPECT_EQ(pool.activeCount(), 0u);
    EXPECT_NE(pool.construct(false), nullptr);
    EXPECT_NE(pool.construct(false), nullptr);
    EXPECT_EQ(pool.activeCount(), 2u);
}

TEST(TypedPool, TrivialTypesWorkWithMoveAndReset) {
    struct TrivialVec {
        float x;
        float y;
        float z;
    };

    TypedPool<TrivialVec> pool(4);
    auto* v1 = pool.construct(1.0f, 2.0f, 3.0f);
    auto* v2 = pool.construct(4.0f, 5.0f, 6.0f);
    ASSERT_NE(v1, nullptr);
    ASSERT_NE(v2, nullptr);
    EXPECT_FLOAT_EQ(v1->y, 2.0f);
    EXPECT_FLOAT_EQ(v2->z, 6.0f);
    EXPECT_EQ(pool.activeCount(), 2u);

    TypedPool<TrivialVec> moved(std::move(pool));
    EXPECT_EQ(moved.activeCount(), 2u);
    EXPECT_TRUE(moved.owns(v1));
    moved.destroy(v1);
    EXPECT_EQ(moved.activeCount(), 1u);
    moved.reset();
    EXPECT_EQ(moved.activeCount(), 0u);
}

TEST(TypedPool, AlignedTypesRespectAlignment) {
    struct alignas(128) AlignedPayload {
        std::uint64_t values[16]{};
    };

    TypedPool<AlignedPayload> pool(4);
    EXPECT_EQ(pool.slotSize() % 128u, 0u);
    for (int i = 0; i < 4; ++i) {
        AlignedPayload* obj = pool.construct();
        ASSERT_NE(obj, nullptr);
        EXPECT_TRUE(isAligned(obj, 128));
    }
}

// -----------------------------------------------------------------------------
// TlsfAllocator Tests
// -----------------------------------------------------------------------------

TEST(TlsfAllocator, InitialStatsAndLargestAllocation) {
    TlsfAllocator tlsf(4096);
    const auto s = tlsf.stats();
    EXPECT_EQ(s.totalBytes, 4096u);
    EXPECT_EQ(s.usedBytes, 0u);
    EXPECT_EQ(s.freeBytes, 4096u);
    EXPECT_EQ(s.allocationCount, 0u);
    EXPECT_GT(s.largestFreeBlock, 4000u);

    void* full = tlsf.allocate(s.largestFreeBlock);
    ASSERT_NE(full, nullptr);
    EXPECT_TRUE(tlsf.owns(full));
    EXPECT_EQ(tlsf.stats().allocationCount, 1u);
    EXPECT_EQ(tlsf.stats().largestFreeBlock, 0u);

    tlsf.deallocate(full);
    const auto after = tlsf.stats();
    EXPECT_EQ(after.usedBytes, 0u);
    EXPECT_EQ(after.freeBytes, 4096u);
    EXPECT_EQ(after.largestFreeBlock, s.largestFreeBlock);
    EXPECT_EQ(after.allocationCount, 0u);
}

TEST(TlsfAllocator, SplittingAndAlignments16_64_128) {
    TlsfAllocator tlsf(16384);
    const std::size_t initialLargest = tlsf.stats().largestFreeBlock;

    std::vector<LiveRegion> live;
    const std::array<std::pair<std::size_t, std::size_t>, 9> requests{{
        {17, 16},
        {33, 64},
        {95, 128},
        {16, 16},
        {64, 64},
        {128, 128},
        {511, 16},
        {1025, 64},
        {2048, 128},
    }};

    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto [sz, align] = requests[i];
        void* p = tlsf.allocate(sz, align);
        ASSERT_NE(p, nullptr) << "Failed at request " << i;
        EXPECT_TRUE(isAligned(p, align));
        EXPECT_TRUE(tlsf.owns(p));

        const auto pattern = static_cast<std::uint8_t>(0x10u + i);
        std::memset(p, pattern, sz);
        const auto addr = reinterpret_cast<std::uintptr_t>(p);
        live.push_back({addr, addr + sz, p, sz, pattern});
    }

    verifyNoOverlapsAndPatterns(live);
    const auto mid = tlsf.stats();
    EXPECT_EQ(mid.allocationCount, requests.size());
    EXPECT_EQ(mid.usedBytes + mid.freeBytes, mid.totalBytes);

    for (const auto& r : live) {
        tlsf.deallocate(r.ptr);
    }

    const auto end = tlsf.stats();
    EXPECT_EQ(end.allocationCount, 0u);
    EXPECT_EQ(end.usedBytes, 0u);
    EXPECT_EQ(end.freeBytes, end.totalBytes);
    EXPECT_EQ(end.largestFreeBlock, initialLargest);
}

TEST(TlsfAllocator, NeighborCoalescingLeftRightAndBoth) {
    constexpr std::size_t kPoolSize = 8192;
    TlsfAllocator tlsf(kPoolSize);
    const std::size_t initialLargest = tlsf.stats().largestFreeBlock;

    // Allocate 4 contiguous blocks [A][B][C][D_anchor] + remainder.
    void* a = tlsf.allocate(512, 16);
    void* b = tlsf.allocate(512, 16);
    void* c = tlsf.allocate(512, 16);
    void* d = tlsf.allocate(512, 16);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(d, nullptr);

    // Exhaust the tail remainder so A/B/C coalescing is isolated from the tail.
    const std::size_t tailSize = tlsf.stats().largestFreeBlock;
    void* tail = tlsf.allocate(tailSize, 16);
    ASSERT_NE(tail, nullptr);
    EXPECT_EQ(tlsf.stats().largestFreeBlock, 0u);

    // 1. Left coalescing: free A, then free B -> AB coalesces to >= 1024 + header.
    tlsf.deallocate(a);
    tlsf.deallocate(b);
    EXPECT_GE(tlsf.stats().largestFreeBlock, 1024u);
    void* mergedAB = tlsf.allocate(1024, 16);
    ASSERT_NE(mergedAB, nullptr);
    EXPECT_EQ(mergedAB, a);

    // Re-split into A and B.
    tlsf.deallocate(mergedAB);
    a = tlsf.allocate(512, 16);
    b = tlsf.allocate(512, 16);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    // 2. Right coalescing: free C, then free B -> BC coalesces to >= 1024 + header.
    tlsf.deallocate(c);
    tlsf.deallocate(b);
    EXPECT_GE(tlsf.stats().largestFreeBlock, 1024u);
    void* mergedBC = tlsf.allocate(1024, 16);
    ASSERT_NE(mergedBC, nullptr);
    EXPECT_EQ(mergedBC, b);

    // Re-split into B and C.
    tlsf.deallocate(mergedBC);
    b = tlsf.allocate(512, 16);
    c = tlsf.allocate(512, 16);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);

    // 3. Both-sides coalescing: free A, free C, then free B -> ABC coalesces to >= 1536.
    tlsf.deallocate(a);
    tlsf.deallocate(c);
    EXPECT_LT(tlsf.stats().largestFreeBlock, 1024u);
    tlsf.deallocate(b);
    EXPECT_GE(tlsf.stats().largestFreeBlock, 1536u);
    void* mergedABC = tlsf.allocate(1536, 16);
    ASSERT_NE(mergedABC, nullptr);
    EXPECT_EQ(mergedABC, a);

    // Free everything and verify complete recovery including tail.
    tlsf.deallocate(mergedABC);
    tlsf.deallocate(d);
    tlsf.deallocate(tail);

    const auto finalStats = tlsf.stats();
    EXPECT_EQ(finalStats.allocationCount, 0u);
    EXPECT_EQ(finalStats.usedBytes, 0u);
    EXPECT_EQ(finalStats.freeBytes, kPoolSize);
    EXPECT_EQ(finalStats.largestFreeBlock, initialLargest);
}

TEST(TlsfAllocator, InvalidInputsAndExhaustion) {
    TlsfAllocator tlsf(1024);
    EXPECT_EQ(tlsf.allocate(0, 16), nullptr);
    EXPECT_EQ(tlsf.allocate(16, 0), nullptr);
    EXPECT_EQ(tlsf.allocate(16, 12), nullptr);
    EXPECT_EQ(tlsf.allocate(2048, 16), nullptr);

    int stackVar = 42;
    EXPECT_FALSE(tlsf.owns(nullptr));
    EXPECT_FALSE(tlsf.owns(&stackVar));

    void* p = tlsf.allocate(64, 16);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(tlsf.owns(p));
    EXPECT_FALSE(tlsf.owns(static_cast<std::byte*>(p) + 4)); // unaligned interior pointer

    tlsf.deallocate(nullptr);
    tlsf.deallocate(p);
    EXPECT_EQ(tlsf.stats().allocationCount, 0u);
    EXPECT_EQ(tlsf.stats().usedBytes, 0u);

    TlsfAllocator tiny(16);
    EXPECT_EQ(tiny.capacity(), 0u);
    EXPECT_EQ(tiny.allocate(16), nullptr);
}

TEST(TlsfAllocator, RandomizedStressNoOverlapsAndFullRecovery) {
    constexpr std::size_t kPoolSize = 2ull * 1024ull * 1024ull; // 2 MB
    TlsfAllocator tlsf(kPoolSize);
    const std::size_t initialLargest = tlsf.stats().largestFreeBlock;

    std::mt19937_64 rng(0x5A17C0FFEEULL);
    const std::array<std::size_t, 7> alignments{1, 8, 16, 32, 64, 128, 256};

    std::vector<LiveRegion> live;
    live.reserve(512);

    std::uint8_t nextPattern = 1;
    for (int step = 0; step < 5000; ++step) {
        const bool shouldFree = !live.empty() && (live.size() > 256 || (rng() % 100) < 48);
        if (shouldFree) {
            const std::size_t idx = static_cast<std::size_t>(rng() % live.size());
            const LiveRegion target = live[idx];
            const auto* bytes = static_cast<const std::uint8_t*>(target.ptr);
            for (std::size_t i = 0; i < target.size; ++i) {
                ASSERT_EQ(bytes[i], target.pattern);
            }
            tlsf.deallocate(target.ptr);
            live[idx] = live.back();
            live.pop_back();
        } else {
            // Mix tiny, medium, and large sizes across multiple TLSF FLI/SLI classes.
            const std::uint64_t bucket = rng() % 10;
            std::size_t reqSize = 1;
            if (bucket < 4) {
                reqSize = 1 + static_cast<std::size_t>(rng() % 128);
            } else if (bucket < 8) {
                reqSize = 129 + static_cast<std::size_t>(rng() % 4096);
            } else {
                reqSize = 4097 + static_cast<std::size_t>(rng() % 32768);
            }
            const std::size_t align = alignments[static_cast<std::size_t>(rng() % alignments.size())];

            void* p = tlsf.allocate(reqSize, align);
            if (p != nullptr) {
                ASSERT_TRUE(isAligned(p, align));
                ASSERT_TRUE(tlsf.owns(p));

                const std::uint8_t pattern = nextPattern++;
                if (nextPattern == 0) {
                    nextPattern = 1;
                }
                std::memset(p, pattern, reqSize);
                const auto addr = reinterpret_cast<std::uintptr_t>(p);
                live.push_back({addr, addr + reqSize, p, reqSize, pattern});
            }
        }

        if ((step % 250) == 0) {
            verifyNoOverlapsAndPatterns(live);
            const auto s = tlsf.stats();
            EXPECT_EQ(s.allocationCount, live.size());
            EXPECT_EQ(s.usedBytes + s.freeBytes, s.totalBytes);
        }
    }

    verifyNoOverlapsAndPatterns(live);

    // Free all remaining allocations in shuffled order.
    std::shuffle(live.begin(), live.end(), rng);
    for (const auto& r : live) {
        tlsf.deallocate(r.ptr);
    }
    live.clear();

    const auto recovered = tlsf.stats();
    EXPECT_EQ(recovered.allocationCount, 0u);
    EXPECT_EQ(recovered.usedBytes, 0u);
    EXPECT_EQ(recovered.freeBytes, kPoolSize);
    EXPECT_EQ(recovered.largestFreeBlock, initialLargest);

    void* fullBlock = tlsf.allocate(initialLargest);
    ASSERT_NE(fullBlock, nullptr);
    tlsf.deallocate(fullBlock);
}

TEST(TlsfAllocator, ResetAndMoveSemantics) {
    TlsfAllocator a(8192);
    void* p = a.allocate(1024, 64);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(a.stats().allocationCount, 1u);

    TlsfAllocator b(std::move(a));
    EXPECT_EQ(a.capacity(), 0u);
    EXPECT_EQ(a.allocate(64), nullptr);
    EXPECT_TRUE(b.owns(p));
    EXPECT_EQ(b.stats().allocationCount, 1u);

    b.reset();
    EXPECT_EQ(b.stats().allocationCount, 0u);
    EXPECT_EQ(b.stats().usedBytes, 0u);
    EXPECT_EQ(b.stats().freeBytes, 8192u);
}

TEST(TlsfAllocator, SmallFrontGapAdvancementAndTailCoalescing) {
    TlsfAllocator tlsf(4096);
    const std::size_t initialLargest = tlsf.stats().largestFreeBlock;

    // Force a 16-byte front gap relative to 64-byte alignment on the remaining block.
    void* p1 = tlsf.allocate(16, 16);
    ASSERT_NE(p1, nullptr);

    void* p2 = tlsf.allocate(64, 64);
    ASSERT_NE(p2, nullptr);
    EXPECT_TRUE(isAligned(p2, 64));

    void* p3 = tlsf.allocate(128, 128);
    ASSERT_NE(p3, nullptr);
    EXPECT_TRUE(isAligned(p3, 128));

    // Free in reverse order so each free coalesces with the trailing isLast() block.
    tlsf.deallocate(p3);
    tlsf.deallocate(p2);
    tlsf.deallocate(p1);

    const auto s = tlsf.stats();
    EXPECT_EQ(s.allocationCount, 0u);
    EXPECT_EQ(s.usedBytes, 0u);
    EXPECT_EQ(s.freeBytes, 4096u);
    EXPECT_EQ(s.largestFreeBlock, initialLargest);
}

TEST(TlsfAllocator, DefaultCapacitySupportsLargeBlocks) {
    TlsfAllocator tlsf;
    EXPECT_EQ(tlsf.capacity(), TlsfAllocator::kDefaultCapacity);

    void* bigA = tlsf.allocate(16ull * 1024ull * 1024ull, 64);
    void* bigB = tlsf.allocate(8ull * 1024ull * 1024ull, 128);
    ASSERT_NE(bigA, nullptr);
    ASSERT_NE(bigB, nullptr);
    EXPECT_TRUE(isAligned(bigA, 64));
    EXPECT_TRUE(isAligned(bigB, 128));
    EXPECT_EQ(tlsf.stats().allocationCount, 2u);

    tlsf.deallocate(bigA);
    tlsf.deallocate(bigB);
    EXPECT_EQ(tlsf.stats().usedBytes, 0u);
    EXPECT_EQ(tlsf.stats().freeBytes, tlsf.capacity());
}
