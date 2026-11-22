#include "core/MemoryAllocators.hpp"

#include "core/Assert.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace lumora {

// -----------------------------------------------------------------------------
// LinearAllocator
// -----------------------------------------------------------------------------

LinearAllocator::LinearAllocator(std::size_t capacity, std::size_t baseAlignment)
    : capacity_(capacity) {
    if (baseAlignment == 0) {
        baseAlignment = kDefaultAlignment;
    }
    if (!std::has_single_bit(baseAlignment)) {
        baseAlignment = std::bit_ceil(baseAlignment);
    }
    baseAlignment_ = (std::max)(baseAlignment, alignof(std::max_align_t));
    if (capacity_ > 0) {
        buffer_ = static_cast<std::byte*>(::operator new(capacity_, std::align_val_t{baseAlignment_}));
    }
}

LinearAllocator::~LinearAllocator() { release(); }

LinearAllocator::LinearAllocator(LinearAllocator&& other) noexcept
    : buffer_(other.buffer_)
    , capacity_(other.capacity_)
    , baseAlignment_(other.baseAlignment_)
    , offset_(other.offset_)
    , peak_(other.peak_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
    other.peak_ = 0;
}

LinearAllocator& LinearAllocator::operator=(LinearAllocator&& other) noexcept {
    if (this != &other) {
        release();
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        baseAlignment_ = other.baseAlignment_;
        offset_ = other.offset_;
        peak_ = other.peak_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
        other.peak_ = 0;
    }
    return *this;
}

void LinearAllocator::release() noexcept {
    if (buffer_ != nullptr) {
        ::operator delete(buffer_, std::align_val_t{baseAlignment_});
        buffer_ = nullptr;
    }
    capacity_ = 0;
    offset_ = 0;
    peak_ = 0;
}

void* LinearAllocator::allocate(std::size_t size, std::size_t align) noexcept {
    if (buffer_ == nullptr || size == 0 || align == 0 || !std::has_single_bit(align)) {
        return nullptr;
    }
    const std::uintptr_t baseAddr = reinterpret_cast<std::uintptr_t>(buffer_);
    const std::uintptr_t currAddr = baseAddr + offset_;
    const std::uintptr_t mask = static_cast<std::uintptr_t>(align - 1u);
    if (currAddr > (std::numeric_limits<std::uintptr_t>::max)() - mask) {
        return nullptr;
    }
    const std::uintptr_t alignedAddr = (currAddr + mask) & ~mask;
    const std::size_t newOffset = static_cast<std::size_t>(alignedAddr - baseAddr);
    if (newOffset > capacity_ || size > capacity_ - newOffset) {
        return nullptr;
    }
    offset_ = newOffset + size;
    peak_ = (std::max)(peak_, offset_);
    return reinterpret_cast<void*>(alignedAddr);
}

void LinearAllocator::rewind(Marker marker) noexcept {
    if (marker <= offset_) {
        offset_ = marker;
    }
}

void LinearAllocator::reset() noexcept { offset_ = 0; }

bool LinearAllocator::owns(const void* ptr) const noexcept {
    if (buffer_ == nullptr || ptr == nullptr) {
        return false;
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer_);
    return addr >= begin && addr < begin + capacity_;
}

// -----------------------------------------------------------------------------
// PoolAllocator
// -----------------------------------------------------------------------------

PoolAllocator::PoolAllocator(std::size_t slotSize, std::size_t slotCount, std::size_t alignment)
    : slotCount_(slotCount) {
    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }
    if (!std::has_single_bit(alignment)) {
        alignment = std::bit_ceil(alignment);
    }
    alignment_ = (std::max)(alignment, alignof(FreeNode));

    const std::size_t minSlot = (std::max)(slotSize, sizeof(FreeNode));
    const std::size_t mask = alignment_ - 1u;
    if (minSlot > (std::numeric_limits<std::size_t>::max)() - mask) {
        throw std::bad_alloc();
    }
    slotStride_ = (minSlot + mask) & ~mask;

    if (slotCount_ > 0) {
        if (slotStride_ > (std::numeric_limits<std::size_t>::max)() / slotCount_) {
            throw std::bad_alloc();
        }
        buffer_ = static_cast<std::byte*>(
            ::operator new(slotStride_ * slotCount_, std::align_val_t{alignment_}));
        buildFreeList();
    }
}

PoolAllocator::~PoolAllocator() { release(); }

PoolAllocator::PoolAllocator(PoolAllocator&& other) noexcept
    : buffer_(other.buffer_)
    , freeHead_(other.freeHead_)
    , slotStride_(other.slotStride_)
    , slotCount_(other.slotCount_)
    , alignment_(other.alignment_)
    , activeCount_(other.activeCount_) {
    other.buffer_ = nullptr;
    other.freeHead_ = nullptr;
    other.slotCount_ = 0;
    other.activeCount_ = 0;
}

PoolAllocator& PoolAllocator::operator=(PoolAllocator&& other) noexcept {
    if (this != &other) {
        release();
        buffer_ = other.buffer_;
        freeHead_ = other.freeHead_;
        slotStride_ = other.slotStride_;
        slotCount_ = other.slotCount_;
        alignment_ = other.alignment_;
        activeCount_ = other.activeCount_;
        other.buffer_ = nullptr;
        other.freeHead_ = nullptr;
        other.slotCount_ = 0;
        other.activeCount_ = 0;
    }
    return *this;
}

void PoolAllocator::release() noexcept {
    if (buffer_ != nullptr) {
        ::operator delete(buffer_, std::align_val_t{alignment_});
        buffer_ = nullptr;
    }
    freeHead_ = nullptr;
    slotCount_ = 0;
    activeCount_ = 0;
}

void PoolAllocator::buildFreeList() noexcept {
    freeHead_ = nullptr;
    activeCount_ = 0;
    if (buffer_ == nullptr || slotCount_ == 0) {
        return;
    }
    for (std::size_t i = slotCount_; i > 0; --i) {
        std::byte* slot = buffer_ + (i - 1u) * slotStride_;
        freeHead_ = ::new (slot) FreeNode{freeHead_};
    }
}

void* PoolAllocator::allocate() noexcept {
    if (freeHead_ == nullptr) {
        return nullptr;
    }
    FreeNode* node = freeHead_;
    freeHead_ = node->next;
    ++activeCount_;
    return static_cast<void*>(node);
}

void PoolAllocator::deallocate(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    LUMORA_ASSERT(owns(ptr), "PoolAllocator::deallocate: pointer is not owned by this pool");
    if (!owns(ptr)) {
        return;
    }
    freeHead_ = ::new (ptr) FreeNode{freeHead_};
    --activeCount_;
}

void PoolAllocator::reset() noexcept { buildFreeList(); }

bool PoolAllocator::owns(const void* ptr) const noexcept {
    if (buffer_ == nullptr || ptr == nullptr || slotCount_ == 0) {
        return false;
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer_);
    const auto end = begin + slotStride_ * slotCount_;
    if (addr < begin || addr >= end) {
        return false;
    }
    return ((addr - begin) % slotStride_) == 0;
}

std::size_t PoolAllocator::slotIndex(const void* ptr) const noexcept {
    if (!owns(ptr)) {
        return slotCount_;
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer_);
    return static_cast<std::size_t>((addr - begin) / slotStride_);
}

void* PoolAllocator::slotPointer(std::size_t index) noexcept {
    if (buffer_ == nullptr || index >= slotCount_) {
        return nullptr;
    }
    return buffer_ + index * slotStride_;
}

const void* PoolAllocator::slotPointer(std::size_t index) const noexcept {
    if (buffer_ == nullptr || index >= slotCount_) {
        return nullptr;
    }
    return buffer_ + index * slotStride_;
}

// -----------------------------------------------------------------------------
// TlsfAllocator
// -----------------------------------------------------------------------------

TlsfAllocator::TlsfAllocator(std::size_t capacityBytes) {
    std::size_t usable = capacityBytes & ~(kBlockAlignment - 1u);
    if (usable > kMaxBlockPayload) {
        usable = kMaxBlockPayload & ~(kBlockAlignment - 1u);
    }
    if (usable < kMinBlockSize) {
        return;
    }
    totalBytes_ = usable;
    buffer_ = static_cast<std::byte*>(
        ::operator new(totalBytes_, std::align_val_t{kBufferAlignment}));
    initPool();
}

TlsfAllocator::~TlsfAllocator() { release(); }

TlsfAllocator::TlsfAllocator(TlsfAllocator&& other) noexcept
    : buffer_(other.buffer_)
    , totalBytes_(other.totalBytes_)
    , usedBytes_(other.usedBytes_)
    , freeBytes_(other.freeBytes_)
    , allocationCount_(other.allocationCount_)
    , flBitmap_(other.flBitmap_) {
    std::memcpy(slBitmap_, other.slBitmap_, sizeof(slBitmap_));
    std::memcpy(freeLists_, other.freeLists_, sizeof(freeLists_));
    other.buffer_ = nullptr;
    other.totalBytes_ = 0;
    other.usedBytes_ = 0;
    other.freeBytes_ = 0;
    other.allocationCount_ = 0;
    other.flBitmap_ = 0;
    std::memset(other.slBitmap_, 0, sizeof(other.slBitmap_));
    std::memset(other.freeLists_, 0, sizeof(other.freeLists_));
}

TlsfAllocator& TlsfAllocator::operator=(TlsfAllocator&& other) noexcept {
    if (this != &other) {
        release();
        buffer_ = other.buffer_;
        totalBytes_ = other.totalBytes_;
        usedBytes_ = other.usedBytes_;
        freeBytes_ = other.freeBytes_;
        allocationCount_ = other.allocationCount_;
        flBitmap_ = other.flBitmap_;
        std::memcpy(slBitmap_, other.slBitmap_, sizeof(slBitmap_));
        std::memcpy(freeLists_, other.freeLists_, sizeof(freeLists_));

        other.buffer_ = nullptr;
        other.totalBytes_ = 0;
        other.usedBytes_ = 0;
        other.freeBytes_ = 0;
        other.allocationCount_ = 0;
        other.flBitmap_ = 0;
        std::memset(other.slBitmap_, 0, sizeof(other.slBitmap_));
        std::memset(other.freeLists_, 0, sizeof(other.freeLists_));
    }
    return *this;
}

void TlsfAllocator::release() noexcept {
    if (buffer_ != nullptr) {
        ::operator delete(buffer_, std::align_val_t{kBufferAlignment});
        buffer_ = nullptr;
    }
    totalBytes_ = 0;
    usedBytes_ = 0;
    freeBytes_ = 0;
    allocationCount_ = 0;
    flBitmap_ = 0;
    std::memset(slBitmap_, 0, sizeof(slBitmap_));
    std::memset(freeLists_, 0, sizeof(freeLists_));
}

void TlsfAllocator::initPool() noexcept {
    flBitmap_ = 0;
    std::memset(slBitmap_, 0, sizeof(slBitmap_));
    std::memset(freeLists_, 0, sizeof(freeLists_));
    usedBytes_ = 0;
    freeBytes_ = 0;
    allocationCount_ = 0;

    if (buffer_ == nullptr || totalBytes_ < kMinBlockSize) {
        return;
    }

    BlockHeader* initial = ::new (buffer_) BlockHeader{};
    const std::size_t initialPayload = totalBytes_ - kHeaderSize;
    initial->prevPhysical = nullptr;
    initial->setPayloadAndFlags(initialPayload, true, true);
    ::new (payloadPtr(initial)) FreeLinks{};
    freeBytes_ = totalBytes_;
    insertFreeBlock(initial);
}

void TlsfAllocator::reset() noexcept { initPool(); }

TlsfAllocator::BinIndex TlsfAllocator::mappingInsert(std::size_t size) noexcept {
    if (size < (std::size_t{1} << kSmallBlockLog2)) {
        return {0u, static_cast<unsigned>(size >> (kSmallBlockLog2 - kSliLog2))};
    }
    const unsigned msb = static_cast<unsigned>(std::bit_width(size) - 1);
    const unsigned fl = msb - (kSmallBlockLog2 - 1u);
    const unsigned sl =
        static_cast<unsigned>((size >> (msb - kSliLog2)) ^ (std::size_t{1} << kSliLog2));
    return {fl, sl};
}

bool TlsfAllocator::mappingSearch(std::size_t size, BinIndex& outBin) noexcept {
    if (size > kMaxBlockPayload) {
        return false;
    }
    if (size >= (std::size_t{1} << kSmallBlockLog2)) {
        const unsigned msb = static_cast<unsigned>(std::bit_width(size) - 1);
        const std::size_t roundMask = (std::size_t{1} << (msb - kSliLog2)) - 1u;
        if (size > kMaxBlockPayload - roundMask) {
            return false;
        }
        size += roundMask;
    }
    outBin = mappingInsert(size);
    return outBin.fl < kFliCount;
}

void TlsfAllocator::insertFreeBlock(BlockHeader* block) noexcept {
    const BinIndex bin = mappingInsert(block->payloadSize());
    LUMORA_ASSERT(bin.fl < kFliCount && bin.sl < kSliCount, "TLSF bin index out of range");

    BlockHeader* head = freeLists_[bin.fl][bin.sl];
    FreeLinks* links = freeLinks(block);
    links->prevFree = nullptr;
    links->nextFree = head;
    if (head != nullptr) {
        freeLinks(head)->prevFree = block;
    }
    freeLists_[bin.fl][bin.sl] = block;
    flBitmap_ |= (std::uint32_t{1} << bin.fl);
    slBitmap_[bin.fl] |= (std::uint32_t{1} << bin.sl);
}

void TlsfAllocator::removeFreeBlock(BlockHeader* block) noexcept {
    const BinIndex bin = mappingInsert(block->payloadSize());
    LUMORA_ASSERT(bin.fl < kFliCount && bin.sl < kSliCount, "TLSF bin index out of range");

    FreeLinks* links = freeLinks(block);
    BlockHeader* prev = links->prevFree;
    BlockHeader* next = links->nextFree;
    if (next != nullptr) {
        freeLinks(next)->prevFree = prev;
    }
    if (prev != nullptr) {
        freeLinks(prev)->nextFree = next;
    } else {
        freeLists_[bin.fl][bin.sl] = next;
        if (next == nullptr) {
            slBitmap_[bin.fl] &= ~(std::uint32_t{1} << bin.sl);
            if (slBitmap_[bin.fl] == 0) {
                flBitmap_ &= ~(std::uint32_t{1} << bin.fl);
            }
        }
    }
    links->prevFree = nullptr;
    links->nextFree = nullptr;
}

TlsfAllocator::BlockHeader* TlsfAllocator::findSuitableBlock(unsigned fl, unsigned sl) const noexcept {
    if (fl >= kFliCount || sl >= kSliCount) {
        return nullptr;
    }
    const std::uint32_t slMap = slBitmap_[fl] & (~std::uint32_t{0} << sl);
    if (slMap != 0) {
        const unsigned foundSl = static_cast<unsigned>(std::countr_zero(slMap));
        return freeLists_[fl][foundSl];
    }
    if (fl + 1u >= kFliCount) {
        return nullptr;
    }
    const std::uint32_t flMap = flBitmap_ & (~std::uint32_t{0} << (fl + 1u));
    if (flMap == 0) {
        return nullptr;
    }
    const unsigned foundFl = static_cast<unsigned>(std::countr_zero(flMap));
    const unsigned foundSl = static_cast<unsigned>(std::countr_zero(slBitmap_[foundFl]));
    return freeLists_[foundFl][foundSl];
}

TlsfAllocator::BlockHeader* TlsfAllocator::findFallbackExactBlock(std::size_t effectiveSize) const noexcept {
    if (effectiveSize > kMaxBlockPayload) {
        return nullptr;
    }
    const BinIndex exact = mappingInsert(effectiveSize);
    if (exact.fl >= kFliCount || (slBitmap_[exact.fl] & (std::uint32_t{1} << exact.sl)) == 0) {
        return nullptr;
    }
    for (BlockHeader* block = freeLists_[exact.fl][exact.sl];
         block != nullptr;
         block = freeLinks(block)->nextFree) {
        if (block->payloadSize() >= effectiveSize) {
            return block;
        }
    }
    return nullptr;
}

bool TlsfAllocator::blockFitsAligned(const BlockHeader* block,
                                       std::size_t effectiveSize,
                                       std::size_t align,
                                       std::size_t& outGap) noexcept {
    const std::uintptr_t payloadAddr = reinterpret_cast<std::uintptr_t>(payloadPtr(block));
    const std::uintptr_t mask = static_cast<std::uintptr_t>(align - 1u);
    if (payloadAddr > (std::numeric_limits<std::uintptr_t>::max)() - mask) {
        return false;
    }
    const std::uintptr_t alignedAddr = (payloadAddr + mask) & ~mask;
    std::size_t gap = static_cast<std::size_t>(alignedAddr - payloadAddr);
    if (gap > 0 && gap < kMinBlockSize) {
        if (align > (std::numeric_limits<std::size_t>::max)() - gap) {
            return false;
        }
        gap += align;
    }
    const std::size_t payload = block->payloadSize();
    if (gap > payload || effectiveSize > payload - gap) {
        return false;
    }
    outGap = gap;
    return true;
}

TlsfAllocator::BlockHeader* TlsfAllocator::findAlignedBlock(std::size_t effectiveSize,
                                                            std::size_t align,
                                                            std::size_t& outGap) const noexcept {
    if (effectiveSize > kMaxBlockPayload) {
        return nullptr;
    }
    const BinIndex start = mappingInsert(effectiveSize);
    if (start.fl >= kFliCount) {
        return nullptr;
    }
    std::uint32_t flMask = flBitmap_ & (~std::uint32_t{0} << start.fl);
    while (flMask != 0) {
        const unsigned fl = static_cast<unsigned>(std::countr_zero(flMask));
        flMask &= (flMask - 1u);

        std::uint32_t slMask = slBitmap_[fl];
        if (fl == start.fl) {
            slMask &= (~std::uint32_t{0} << start.sl);
        }
        while (slMask != 0) {
            const unsigned sl = static_cast<unsigned>(std::countr_zero(slMask));
            slMask &= (slMask - 1u);
            for (BlockHeader* block = freeLists_[fl][sl];
                 block != nullptr;
                 block = freeLinks(block)->nextFree) {
                std::size_t gap = 0;
                if (blockFitsAligned(block, effectiveSize, align, gap)) {
                    outGap = gap;
                    return block;
                }
            }
        }
    }
    return nullptr;
}

TlsfAllocator::BlockHeader* TlsfAllocator::prepareAllocatedBlock(BlockHeader* block,
                                                                 std::size_t effectiveSize,
                                                                 std::size_t frontGap) noexcept {
    removeFreeBlock(block);

    const bool originalIsLast = block->isLast();
    const std::size_t originalPayload = block->payloadSize();

    BlockHeader* allocBlock = block;
    if (frontGap > 0) {
        LUMORA_ASSERT(frontGap >= kMinBlockSize, "Front alignment gap must fit a minimum free block");
        const std::size_t frontPayload = frontGap - kHeaderSize;
        block->setPayloadAndFlags(frontPayload, true, false);
        ::new (payloadPtr(block)) FreeLinks{};
        insertFreeBlock(block);

        allocBlock = ::new (reinterpret_cast<std::byte*>(block) + frontGap) BlockHeader{};
        allocBlock->prevPhysical = block;
        allocBlock->setPayloadAndFlags(originalPayload - frontGap, false, originalIsLast);
        if (!originalIsLast) {
            allocBlock->nextPhysical()->prevPhysical = allocBlock;
        }
    } else {
        allocBlock->setFree(false);
    }

    const std::size_t allocPayload = allocBlock->payloadSize();
    const std::size_t remainder = allocPayload - effectiveSize;
    if (remainder >= kMinBlockSize) {
        const bool allocWasLast = allocBlock->isLast();
        allocBlock->setPayloadAndFlags(effectiveSize, false, false);

        BlockHeader* tailBlock =
            ::new (reinterpret_cast<std::byte*>(allocBlock) + kHeaderSize + effectiveSize) BlockHeader{};
        tailBlock->prevPhysical = allocBlock;
        tailBlock->setPayloadAndFlags(remainder - kHeaderSize, true, allocWasLast);
        ::new (payloadPtr(tailBlock)) FreeLinks{};
        if (!allocWasLast) {
            tailBlock->nextPhysical()->prevPhysical = tailBlock;
        }
        insertFreeBlock(tailBlock);
    }

    const std::size_t consumedBytes = kHeaderSize + allocBlock->payloadSize();
    usedBytes_ += consumedBytes;
    freeBytes_ -= consumedBytes;
    ++allocationCount_;
    return allocBlock;
}

void* TlsfAllocator::allocate(std::size_t size, std::size_t align) noexcept {
    if (buffer_ == nullptr || flBitmap_ == 0 || size == 0 || align == 0 || !std::has_single_bit(align)) {
        return nullptr;
    }

    const std::size_t clampedMin = (std::max)(size, kMinPayloadSize);
    const std::size_t alignMask = kBlockAlignment - 1u;
    if (clampedMin > kMaxBlockPayload - alignMask) {
        return nullptr;
    }
    const std::size_t effectiveSize = (clampedMin + alignMask) & ~alignMask;
    if (effectiveSize > freeBytes_) {
        return nullptr;
    }

    if (align <= kBlockAlignment) {
        BinIndex bin{};
        BlockHeader* block = nullptr;
        if (mappingSearch(effectiveSize, bin)) {
            block = findSuitableBlock(bin.fl, bin.sl);
        }
        if (block == nullptr) {
            block = findFallbackExactBlock(effectiveSize);
        }
        if (block == nullptr) {
            return nullptr;
        }
        BlockHeader* allocBlock = prepareAllocatedBlock(block, effectiveSize, 0);
        return payloadPtr(allocBlock);
    }

    BlockHeader* block = nullptr;
    std::size_t frontGap = 0;
    const std::size_t extra = align + kHeaderSize;
    if (extra <= kMaxBlockPayload && effectiveSize <= kMaxBlockPayload - extra) {
        BinIndex bin{};
        if (mappingSearch(effectiveSize + extra, bin)) {
            BlockHeader* candidate = findSuitableBlock(bin.fl, bin.sl);
            if (candidate != nullptr && blockFitsAligned(candidate, effectiveSize, align, frontGap)) {
                block = candidate;
            }
        }
    }
    if (block == nullptr) {
        block = findAlignedBlock(effectiveSize, align, frontGap);
    }
    if (block == nullptr) {
        return nullptr;
    }

    BlockHeader* allocBlock = prepareAllocatedBlock(block, effectiveSize, frontGap);
    return payloadPtr(allocBlock);
}

void TlsfAllocator::deallocate(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    LUMORA_ASSERT(owns(ptr), "TlsfAllocator::deallocate: pointer is not owned by this allocator");
    if (!owns(ptr)) {
        return;
    }

    auto* block = reinterpret_cast<BlockHeader*>(static_cast<std::byte*>(ptr) - kHeaderSize);
    LUMORA_ASSERT(!block->isFree(), "TlsfAllocator::deallocate: double free detected");
    if (block->isFree()) {
        return;
    }

    const std::size_t releasedBytes = kHeaderSize + block->payloadSize();
    usedBytes_ -= releasedBytes;
    freeBytes_ += releasedBytes;
    --allocationCount_;

    block->setFree(true);

    if (block->prevPhysical != nullptr && block->prevPhysical->isFree()) {
        BlockHeader* prev = block->prevPhysical;
        removeFreeBlock(prev);
        const std::size_t mergedPayload = prev->payloadSize() + kHeaderSize + block->payloadSize();
        const bool mergedLast = block->isLast();
        prev->setPayloadAndFlags(mergedPayload, true, mergedLast);
        if (!mergedLast) {
            prev->nextPhysical()->prevPhysical = prev;
        }
        block = prev;
    }

    if (!block->isLast()) {
        BlockHeader* next = block->nextPhysical();
        if (next->isFree()) {
            removeFreeBlock(next);
            const std::size_t mergedPayload = block->payloadSize() + kHeaderSize + next->payloadSize();
            const bool mergedLast = next->isLast();
            block->setPayloadAndFlags(mergedPayload, true, mergedLast);
            if (!mergedLast) {
                block->nextPhysical()->prevPhysical = block;
            }
        }
    }

    ::new (payloadPtr(block)) FreeLinks{};
    insertFreeBlock(block);
}

bool TlsfAllocator::owns(const void* ptr) const noexcept {
    if (buffer_ == nullptr || ptr == nullptr || totalBytes_ < kMinBlockSize) {
        return false;
    }
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer_);
    const auto end = begin + totalBytes_;
    if (addr < begin + kHeaderSize || addr >= end) {
        return false;
    }
    return ((addr - begin) & (kBlockAlignment - 1u)) == 0;
}

TlsfAllocator::Stats TlsfAllocator::stats() const noexcept {
    Stats out;
    out.totalBytes = totalBytes_;
    out.usedBytes = usedBytes_;
    out.freeBytes = freeBytes_;
    out.allocationCount = allocationCount_;
    if (flBitmap_ != 0) {
        const unsigned maxFl = static_cast<unsigned>(std::bit_width(flBitmap_) - 1);
        const unsigned maxSl = static_cast<unsigned>(std::bit_width(slBitmap_[maxFl]) - 1);
        std::size_t largest = 0;
        for (const BlockHeader* block = freeLists_[maxFl][maxSl];
             block != nullptr;
             block = freeLinks(block)->nextFree) {
            largest = (std::max)(largest, block->payloadSize());
        }
        out.largestFreeBlock = largest;
    }
    return out;
}

} // namespace lumora
