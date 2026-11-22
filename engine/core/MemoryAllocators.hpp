#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace lumora {

/// Contiguous O(1) bump-pointer allocator for frame/transient allocations.
/// Supports arbitrary power-of-two alignments, marker rewind, RAII FrameScope,
/// and bulk reset without individual deallocations.
class LinearAllocator {
public:
    using Marker = std::size_t;

    inline static constexpr std::size_t kDefaultCapacity = 64ull * 1024ull * 1024ull; // 64 MB
    inline static constexpr std::size_t kDefaultAlignment = 64;

    explicit LinearAllocator(std::size_t capacity = kDefaultCapacity,
                             std::size_t baseAlignment = kDefaultAlignment);
    ~LinearAllocator();

    LinearAllocator(const LinearAllocator&) = delete;
    LinearAllocator& operator=(const LinearAllocator&) = delete;
    LinearAllocator(LinearAllocator&& other) noexcept;
    LinearAllocator& operator=(LinearAllocator&& other) noexcept;

    /// RAII guard that captures the allocator marker on construction and rewinds on destruction.
    class FrameScope {
    public:
        explicit FrameScope(LinearAllocator& allocator) noexcept
            : allocator_(allocator)
            , marker_(allocator.marker()) {}

        ~FrameScope() { allocator_.rewind(marker_); }

        FrameScope(const FrameScope&) = delete;
        FrameScope& operator=(const FrameScope&) = delete;
        FrameScope(FrameScope&&) = delete;
        FrameScope& operator=(FrameScope&&) = delete;

        [[nodiscard]] void* allocate(std::size_t size,
                                     std::size_t align = alignof(std::max_align_t)) noexcept {
            return allocator_.allocate(size, align);
        }

        template <class T>
        [[nodiscard]] T* allocateArray(std::size_t count, std::size_t align = alignof(T)) noexcept {
            return allocator_.allocateArray<T>(count, align);
        }

        [[nodiscard]] Marker marker() const noexcept { return marker_; }

    private:
        LinearAllocator& allocator_;
        Marker marker_;
    };

    /// Allocates `size` bytes aligned to `align`. Returns nullptr on overflow or invalid arguments.
    [[nodiscard]] void* allocate(std::size_t size,
                                 std::size_t align = alignof(std::max_align_t)) noexcept;

    /// Allocates raw aligned storage for `count` elements of `T`. Returns nullptr on overflow.
    template <class T>
    [[nodiscard]] T* allocateArray(std::size_t count, std::size_t align = alignof(T)) noexcept {
        if (count == 0 || align == 0 || !std::has_single_bit(align)) {
            return nullptr;
        }
        if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
            return nullptr;
        }
        const std::size_t effectiveAlign = (std::max)(align, alignof(T));
        return static_cast<T*>(allocate(count * sizeof(T), effectiveAlign));
    }

    [[nodiscard]] Marker marker() const noexcept { return offset_; }
    void rewind(Marker marker) noexcept;
    void reset() noexcept;

    [[nodiscard]] bool owns(const void* ptr) const noexcept;
    [[nodiscard]] const void* data() const noexcept { return buffer_; }
    [[nodiscard]] std::size_t used() const noexcept { return offset_; }
    [[nodiscard]] std::size_t peak() const noexcept { return peak_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t remaining() const noexcept { return capacity_ - offset_; }

private:
    void release() noexcept;

    std::byte* buffer_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t baseAlignment_ = kDefaultAlignment;
    std::size_t offset_ = 0;
    std::size_t peak_ = 0;
};

using FrameScope = LinearAllocator::FrameScope;

/// Fixed-slot O(1) intrusive free-list allocator.
class PoolAllocator {
public:
    PoolAllocator(std::size_t slotSize,
                  std::size_t slotCount,
                  std::size_t alignment = alignof(std::max_align_t));
    ~PoolAllocator();

    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    PoolAllocator(PoolAllocator&& other) noexcept;
    PoolAllocator& operator=(PoolAllocator&& other) noexcept;

    /// Pops one slot from the free list in O(1). Returns nullptr when exhausted.
    [[nodiscard]] void* allocate() noexcept;

    /// Pushes an owned slot back onto the free list in O(1).
    void deallocate(void* ptr) noexcept;

    /// Rebuilds the free list and resets active count to zero.
    void reset() noexcept;

    [[nodiscard]] std::size_t activeCount() const noexcept { return activeCount_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return slotCount_; }
    [[nodiscard]] std::size_t availableCount() const noexcept { return slotCount_ - activeCount_; }
    [[nodiscard]] std::size_t slotSize() const noexcept { return slotStride_; }
    [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
    [[nodiscard]] bool owns(const void* ptr) const noexcept;

    /// Returns the slot index in [0, capacity()), or capacity() if not owned.
    [[nodiscard]] std::size_t slotIndex(const void* ptr) const noexcept;
    [[nodiscard]] void* slotPointer(std::size_t index) noexcept;
    [[nodiscard]] const void* slotPointer(std::size_t index) const noexcept;

private:
    struct FreeNode {
        FreeNode* next = nullptr;
    };

    void release() noexcept;
    void buildFreeList() noexcept;

    std::byte* buffer_ = nullptr;
    FreeNode* freeHead_ = nullptr;
    std::size_t slotStride_ = 0;
    std::size_t slotCount_ = 0;
    std::size_t alignment_ = alignof(std::max_align_t);
    std::size_t activeCount_ = 0;
};

/// Typed object pool backed by PoolAllocator with lifetime management (construct/destroy).
/// Any non-trivially-destructible live objects remaining on reset() or destruction are destroyed.
template <class T>
class TypedPool {
public:
    explicit TypedPool(std::size_t slotCount, std::size_t alignment = alignof(T))
        : pool_(sizeof(T), slotCount, (std::max)(alignment, alignof(T))) {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            initLiveMask(slotCount);
        }
    }

    ~TypedPool() { destroyAllLive(); }

    TypedPool(const TypedPool&) = delete;
    TypedPool& operator=(const TypedPool&) = delete;

    TypedPool(TypedPool&& other) noexcept
        : pool_(std::move(other.pool_))
        , liveWords_(std::move(other.liveWords_)) {}

    TypedPool& operator=(TypedPool&& other) noexcept {
        if (this != &other) {
            destroyAllLive();
            pool_ = std::move(other.pool_);
            liveWords_ = std::move(other.liveWords_);
        }
        return *this;
    }

    template <class... Args>
    [[nodiscard]] T* construct(Args&&... args) {
        void* mem = pool_.allocate();
        if (!mem) {
            return nullptr;
        }
        T* obj = nullptr;
        if constexpr (std::is_nothrow_constructible_v<T, Args...>) {
            obj = std::construct_at(static_cast<T*>(mem), std::forward<Args>(args)...);
        } else {
            try {
                obj = std::construct_at(static_cast<T*>(mem), std::forward<Args>(args)...);
            } catch (...) {
                pool_.deallocate(mem);
                throw;
            }
        }
        if constexpr (!std::is_trivially_destructible_v<T>) {
            setLive(pool_.slotIndex(obj), true);
        }
        return obj;
    }

    void destroy(T* ptr) noexcept {
        if (!ptr || !pool_.owns(ptr)) {
            return;
        }
        if constexpr (!std::is_trivially_destructible_v<T>) {
            const std::size_t idx = pool_.slotIndex(ptr);
            if (!isLive(idx)) {
                return;
            }
            setLive(idx, false);
        }
        std::destroy_at(ptr);
        pool_.deallocate(ptr);
    }

    void reset() noexcept {
        destroyAllLive();
        pool_.reset();
    }

    [[nodiscard]] std::size_t activeCount() const noexcept { return pool_.activeCount(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return pool_.capacity(); }
    [[nodiscard]] std::size_t availableCount() const noexcept { return pool_.availableCount(); }
    [[nodiscard]] std::size_t slotSize() const noexcept { return pool_.slotSize(); }
    [[nodiscard]] bool owns(const void* ptr) const noexcept { return pool_.owns(ptr); }

private:
    void initLiveMask(std::size_t slotCount) {
        const std::size_t words = (slotCount + 63u) / 64u;
        liveWords_.assign(words, 0);
    }

    [[nodiscard]] bool isLive(std::size_t index) const noexcept {
        const std::size_t word = index / 64u;
        const unsigned bit = static_cast<unsigned>(index % 64u);
        return word < liveWords_.size() && ((liveWords_[word] & (std::uint64_t{1} << bit)) != 0);
    }

    void setLive(std::size_t index, bool live) noexcept {
        const std::size_t word = index / 64u;
        const unsigned bit = static_cast<unsigned>(index % 64u);
        if (word >= liveWords_.size()) {
            return;
        }
        if (live) {
            liveWords_[word] |= (std::uint64_t{1} << bit);
        } else {
            liveWords_[word] &= ~(std::uint64_t{1} << bit);
        }
    }

    void destroyAllLive() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            if (pool_.activeCount() == 0) {
                std::fill(liveWords_.begin(), liveWords_.end(), std::uint64_t{0});
                return;
            }
            for (std::size_t w = 0; w < liveWords_.size(); ++w) {
                while (liveWords_[w] != 0) {
                    const unsigned bit = static_cast<unsigned>(std::countr_zero(liveWords_[w]));
                    liveWords_[w] &= ~(std::uint64_t{1} << bit);
                    const std::size_t idx = w * 64u + bit;
                    T* obj = std::launder(static_cast<T*>(pool_.slotPointer(idx)));
                    std::destroy_at(obj);
                }
            }
        }
    }

    PoolAllocator pool_;
    std::vector<std::uint64_t> liveWords_;
};

/// Two-Level Segregated Fit (TLSF) O(1) variable-size allocator with FLI/SLI bitmaps,
/// boundary tags, block splitting, immediate neighbor coalescing, and arbitrary alignments.
class TlsfAllocator {
public:
    struct Stats {
        std::size_t totalBytes = 0;
        std::size_t usedBytes = 0;
        std::size_t freeBytes = 0;
        std::size_t largestFreeBlock = 0;
        std::size_t allocationCount = 0;
    };

    inline static constexpr std::size_t kDefaultCapacity = 64ull * 1024ull * 1024ull; // 64 MB

    explicit TlsfAllocator(std::size_t capacityBytes = kDefaultCapacity);
    ~TlsfAllocator();

    TlsfAllocator(const TlsfAllocator&) = delete;
    TlsfAllocator& operator=(const TlsfAllocator&) = delete;
    TlsfAllocator(TlsfAllocator&& other) noexcept;
    TlsfAllocator& operator=(TlsfAllocator&& other) noexcept;

    [[nodiscard]] void* allocate(std::size_t size,
                                 std::size_t align = alignof(std::max_align_t)) noexcept;
    void deallocate(void* ptr) noexcept;
    void reset() noexcept;

    [[nodiscard]] bool owns(const void* ptr) const noexcept;
    [[nodiscard]] Stats stats() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return totalBytes_; }

private:
    static constexpr unsigned kBlockAlignLog2 = (sizeof(void*) >= 8) ? 4u : 3u;
    static constexpr std::size_t kBlockAlignment = std::size_t{1} << kBlockAlignLog2;
    static constexpr unsigned kSliLog2 = 5u;
    static constexpr unsigned kSliCount = 1u << kSliLog2; // 32
    static constexpr unsigned kSmallBlockLog2 = kBlockAlignLog2 + kSliLog2;
    static constexpr unsigned kFliCount = (sizeof(std::size_t) >= 8) ? 32u : 24u;
    static constexpr std::size_t kBufferAlignment = 64u;

    struct BlockHeader {
        BlockHeader* prevPhysical = nullptr;
        std::size_t sizeAndFlags = 0;

        static constexpr std::size_t kFreeBit = 1u;
        static constexpr std::size_t kLastBit = 2u;
        static constexpr std::size_t kFlagsMask = kFreeBit | kLastBit;
        static constexpr std::size_t kSizeMask = ~kFlagsMask;

        [[nodiscard]] std::size_t payloadSize() const noexcept { return sizeAndFlags & kSizeMask; }
        [[nodiscard]] bool isFree() const noexcept { return (sizeAndFlags & kFreeBit) != 0; }
        [[nodiscard]] bool isLast() const noexcept { return (sizeAndFlags & kLastBit) != 0; }

        void setPayloadAndFlags(std::size_t size, bool freeFlag, bool lastFlag) noexcept {
            sizeAndFlags = (size & kSizeMask)
                         | (freeFlag ? kFreeBit : std::size_t{0})
                         | (lastFlag ? kLastBit : std::size_t{0});
        }

        void setFree(bool freeFlag) noexcept {
            if (freeFlag) {
                sizeAndFlags |= kFreeBit;
            } else {
                sizeAndFlags &= ~kFreeBit;
            }
        }

        void setLast(bool lastFlag) noexcept {
            if (lastFlag) {
                sizeAndFlags |= kLastBit;
            } else {
                sizeAndFlags &= ~kLastBit;
            }
        }

        [[nodiscard]] BlockHeader* nextPhysical() noexcept {
            return reinterpret_cast<BlockHeader*>(
                reinterpret_cast<std::byte*>(this) + sizeof(BlockHeader) + payloadSize());
        }

        [[nodiscard]] const BlockHeader* nextPhysical() const noexcept {
            return reinterpret_cast<const BlockHeader*>(
                reinterpret_cast<const std::byte*>(this) + sizeof(BlockHeader) + payloadSize());
        }
    };

    struct FreeLinks {
        BlockHeader* nextFree = nullptr;
        BlockHeader* prevFree = nullptr;
    };

    static constexpr std::size_t kHeaderSize = sizeof(BlockHeader);
    static constexpr std::size_t kMinPayloadSize = sizeof(FreeLinks);
    static constexpr std::size_t kMinBlockSize = kHeaderSize + kMinPayloadSize;
    static constexpr std::size_t kMaxBlockPayload =
        (std::size_t{1} << (kFliCount + kSmallBlockLog2 - 1u)) - kBlockAlignment;

    static_assert(sizeof(BlockHeader) == kBlockAlignment);
    static_assert(sizeof(FreeLinks) == kBlockAlignment);
    static_assert(kBlockAlignment >= alignof(std::max_align_t));

    struct BinIndex {
        unsigned fl = 0;
        unsigned sl = 0;
    };

    [[nodiscard]] static void* payloadPtr(BlockHeader* block) noexcept {
        return reinterpret_cast<std::byte*>(block) + kHeaderSize;
    }

    [[nodiscard]] static const void* payloadPtr(const BlockHeader* block) noexcept {
        return reinterpret_cast<const std::byte*>(block) + kHeaderSize;
    }

    [[nodiscard]] static FreeLinks* freeLinks(BlockHeader* block) noexcept {
        return std::launder(reinterpret_cast<FreeLinks*>(payloadPtr(block)));
    }

    [[nodiscard]] static const FreeLinks* freeLinks(const BlockHeader* block) noexcept {
        return std::launder(reinterpret_cast<const FreeLinks*>(payloadPtr(block)));
    }

    [[nodiscard]] static BinIndex mappingInsert(std::size_t size) noexcept;
    [[nodiscard]] static bool mappingSearch(std::size_t size, BinIndex& outBin) noexcept;

    void release() noexcept;
    void initPool() noexcept;

    void insertFreeBlock(BlockHeader* block) noexcept;
    void removeFreeBlock(BlockHeader* block) noexcept;

    [[nodiscard]] BlockHeader* findSuitableBlock(unsigned fl, unsigned sl) const noexcept;
    [[nodiscard]] BlockHeader* findFallbackExactBlock(std::size_t effectiveSize) const noexcept;
    [[nodiscard]] BlockHeader* findAlignedBlock(std::size_t effectiveSize,
                                                std::size_t align,
                                                std::size_t& outGap) const noexcept;
    [[nodiscard]] static bool blockFitsAligned(const BlockHeader* block,
                                               std::size_t effectiveSize,
                                               std::size_t align,
                                               std::size_t& outGap) noexcept;

    [[nodiscard]] BlockHeader* prepareAllocatedBlock(BlockHeader* block,
                                                     std::size_t effectiveSize,
                                                     std::size_t frontGap) noexcept;

    std::byte* buffer_ = nullptr;
    std::size_t totalBytes_ = 0;
    std::size_t usedBytes_ = 0;
    std::size_t freeBytes_ = 0;
    std::size_t allocationCount_ = 0;

    std::uint32_t flBitmap_ = 0;
    std::uint32_t slBitmap_[kFliCount]{};
    BlockHeader* freeLists_[kFliCount][kSliCount]{};
};

} // namespace lumora
