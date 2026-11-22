#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#define LUMORA_SOA_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define LUMORA_SOA_RESTRICT __restrict__
#else
#define LUMORA_SOA_RESTRICT
#endif

namespace lumora {

/// Cache-line alignment (64 bytes) guaranteed for all SoA column allocations.
inline constexpr std::size_t kSoaCacheLineBytes = 64;

/// Propagates a compiler alignment assumption when `ptr` is non-null.
template <std::size_t Align = kSoaCacheLineBytes, class T>
[[nodiscard]] constexpr T* assumeAligned(T* ptr) noexcept {
    static_assert(Align > 0 && (Align & (Align - 1)) == 0, "Alignment must be a positive power of two.");
    return ptr == nullptr ? nullptr : std::assume_aligned<Align>(ptr);
}

/// Returns true when `ptr` is aligned to a 64-byte cache-line boundary.
template <class T>
[[nodiscard]] inline bool isCacheLineAligned(const T* ptr) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) % kSoaCacheLineBytes) == 0;
}

/// Contiguous, 64-byte cache-line aligned storage column for SoA component arrays.
/// Supports O(1) swap-and-pop removal via `swapRemove` and compiler alignment hints via `data()`.
template <class T>
class AlignedSoaColumn {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    static constexpr std::size_t kAlignment =
        alignof(T) > kSoaCacheLineBytes ? alignof(T) : kSoaCacheLineBytes;

    AlignedSoaColumn() noexcept = default;

    explicit AlignedSoaColumn(std::size_t count) {
        resize(count);
    }

    AlignedSoaColumn(std::size_t count, const T& value) {
        resize(count, value);
    }

    ~AlignedSoaColumn() {
        clear();
        deallocateStorage(data_);
        data_ = nullptr;
        capacity_ = 0;
    }

    AlignedSoaColumn(const AlignedSoaColumn& other) {
        if (other.size_ == 0) {
            return;
        }
        T* buffer = allocateStorage(other.size_);
        try {
            std::uninitialized_copy_n(other.data_, other.size_, buffer);
        } catch (...) {
            deallocateStorage(buffer);
            throw;
        }
        data_ = buffer;
        size_ = other.size_;
        capacity_ = other.size_;
    }

    AlignedSoaColumn& operator=(const AlignedSoaColumn& other) {
        if (this == &other) {
            return *this;
        }
        AlignedSoaColumn copy(other);
        swap(copy);
        return *this;
    }

    AlignedSoaColumn(AlignedSoaColumn&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedSoaColumn& operator=(AlignedSoaColumn&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        clear();
        deallocateStorage(data_);
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
        return *this;
    }

    void swap(AlignedSoaColumn& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    void pushBack(const T& value) {
        emplaceBack(value);
    }

    void pushBack(T&& value) {
        emplaceBack(std::move(value));
    }

    template <class... Args>
    T& emplaceBack(Args&&... args) {
        if (size_ == capacity_) {
            const std::size_t newCapacity = nextGrowthCapacity(capacity_);
            T* newData = allocateStorage(newCapacity);
            // Construct the new element before relocating/destroying the old buffer so
            // self-referential inserts such as `col.pushBack(col[0])` remain valid.
            std::construct_at(newData + size_, std::forward<Args>(args)...);
            try {
                relocateOldToNew(newData);
            } catch (...) {
                std::destroy_at(newData + size_);
                deallocateStorage(newData);
                throw;
            }
            std::destroy_n(data_, size_);
            deallocateStorage(data_);
            data_ = newData;
            capacity_ = newCapacity;
        } else {
            std::construct_at(data_ + size_, std::forward<Args>(args)...);
        }
        ++size_;
        return data_[size_ - 1];
    }

    /// Removes the element at `index` in O(1) by moving the back element into its slot.
    void swapRemove(std::size_t index) {
        if (index >= size_) {
            throw std::out_of_range("AlignedSoaColumn::swapRemove: index out of range");
        }
        const std::size_t last = size_ - 1;
        if (index != last) {
            data_[index] = std::move(data_[last]);
        }
        std::destroy_at(data_ + last);
        --size_;
    }

    void reserve(std::size_t newCapacity) {
        if (newCapacity <= capacity_) {
            return;
        }
        reallocate(newCapacity);
    }

    void resize(std::size_t newSize) {
        if (newSize < size_) {
            std::destroy_n(data_ + newSize, size_ - newSize);
            size_ = newSize;
            return;
        }
        if (newSize > size_) {
            if (newSize > capacity_) {
                reallocate(newSize);
            }
            std::uninitialized_value_construct_n(data_ + size_, newSize - size_);
            size_ = newSize;
        }
    }

    void resize(std::size_t newSize, const T& value) {
        if (newSize < size_) {
            std::destroy_n(data_ + newSize, size_ - newSize);
            size_ = newSize;
            return;
        }
        if (newSize > size_) {
            if (newSize > capacity_) {
                reallocate(newSize);
            }
            std::uninitialized_fill_n(data_ + size_, newSize - size_, value);
            size_ = newSize;
        }
    }

    void clear() noexcept {
        if (size_ > 0) {
            std::destroy_n(data_, size_);
            size_ = 0;
        }
    }

    void shrinkToFit() {
        if (size_ == capacity_) {
            return;
        }
        if (size_ == 0) {
            deallocateStorage(data_);
            data_ = nullptr;
            capacity_ = 0;
            return;
        }
        reallocate(size_);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] T* data() noexcept { return assumeAligned<kAlignment>(data_); }
    [[nodiscard]] const T* data() const noexcept { return assumeAligned<kAlignment>(data_); }

    /// Unconditional alignment hint for hot loops where `size() > 0` has already been verified.
    [[nodiscard]] T* assumeAlignedData() noexcept { return std::assume_aligned<kAlignment>(data_); }
    [[nodiscard]] const T* assumeAlignedData() const noexcept { return std::assume_aligned<kAlignment>(data_); }

    [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }
    [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

    [[nodiscard]] T& at(std::size_t index) {
        if (index >= size_) {
            throw std::out_of_range("AlignedSoaColumn::at: index out of range");
        }
        return data_[index];
    }

    [[nodiscard]] const T& at(std::size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("AlignedSoaColumn::at: index out of range");
        }
        return data_[index];
    }

    [[nodiscard]] std::span<T> span() noexcept { return {data(), size_}; }
    [[nodiscard]] std::span<const T> span() const noexcept { return {data(), size_}; }

    [[nodiscard]] T* begin() noexcept { return data(); }
    [[nodiscard]] T* end() noexcept { return data() + size_; }
    [[nodiscard]] const T* begin() const noexcept { return data(); }
    [[nodiscard]] const T* end() const noexcept { return data() + size_; }
    [[nodiscard]] const T* cbegin() const noexcept { return data(); }
    [[nodiscard]] const T* cend() const noexcept { return data() + size_; }

private:
    static T* allocateStorage(std::size_t count) {
        if (count == 0) {
            return nullptr;
        }
        constexpr std::size_t kMaxElems = (std::numeric_limits<std::size_t>::max)() / sizeof(T);
        if (count > kMaxElems) {
            throw std::bad_array_new_length();
        }
        void* raw = ::operator new(count * sizeof(T), std::align_val_t{kAlignment});
        return static_cast<T*>(raw);
    }

    static void deallocateStorage(T* ptr) noexcept {
        if (ptr != nullptr) {
            ::operator delete(ptr, std::align_val_t{kAlignment});
        }
    }

    static constexpr std::size_t nextGrowthCapacity(std::size_t currentCapacity) noexcept {
        constexpr std::size_t kMinElements =
            (kSoaCacheLineBytes / sizeof(T)) > 1 ? (kSoaCacheLineBytes / sizeof(T)) : 1;
        if (currentCapacity == 0) {
            return kMinElements;
        }
        constexpr std::size_t kMaxElems = (std::numeric_limits<std::size_t>::max)() / sizeof(T);
        if (currentCapacity > kMaxElems / 2) {
            return kMaxElems;
        }
        return std::max(kMinElements, currentCapacity * 2);
    }

    void relocateOldToNew(T* newData) {
        if (size_ == 0) {
            return;
        }
        if constexpr (std::is_nothrow_move_constructible_v<T> || !std::is_copy_constructible_v<T>) {
            std::uninitialized_move_n(data_, size_, newData);
        } else {
            std::uninitialized_copy_n(data_, size_, newData);
        }
    }

    void reallocate(std::size_t newCapacity) {
        T* newData = allocateStorage(newCapacity);
        try {
            relocateOldToNew(newData);
        } catch (...) {
            deallocateStorage(newData);
            throw;
        }
        std::destroy_n(data_, size_);
        deallocateStorage(data_);
        data_ = newData;
        capacity_ = newCapacity;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

/// Entity identifier compatible with widened `entt::entity` values or 64-bit `Uuid`s.
using EntityId = std::uint64_t;
inline constexpr EntityId kInvalidEntityId = ~std::uint64_t{0};

/// 64-bit bitmask identifying an archetype's component/tag signature.
using ArchetypeMask = std::uint64_t;

namespace ComponentTag {
inline constexpr ArchetypeMask None         = 0;
inline constexpr ArchetypeMask Position     = std::uint64_t{1} << 0;
inline constexpr ArchetypeMask Velocity     = std::uint64_t{1} << 1;
inline constexpr ArchetypeMask Acceleration = std::uint64_t{1} << 2;
inline constexpr ArchetypeMask Scale        = std::uint64_t{1} << 3;
inline constexpr ArchetypeMask Bounds       = std::uint64_t{1} << 4;

inline constexpr ArchetypeMask Custom0      = std::uint64_t{1} << 5;
inline constexpr ArchetypeMask Custom1      = std::uint64_t{1} << 6;
inline constexpr ArchetypeMask Custom2      = std::uint64_t{1} << 7;
inline constexpr ArchetypeMask Custom3      = std::uint64_t{1} << 8;
inline constexpr ArchetypeMask Custom4      = std::uint64_t{1} << 9;
inline constexpr ArchetypeMask Custom5      = std::uint64_t{1} << 10;
inline constexpr ArchetypeMask Custom6      = std::uint64_t{1} << 11;
inline constexpr ArchetypeMask Custom7      = std::uint64_t{1} << 12;

inline constexpr ArchetypeMask AllKinematic = Position | Velocity | Acceleration | Scale | Bounds;

[[nodiscard]] constexpr ArchetypeMask customBit(std::uint32_t index) noexcept {
    return index < 59u ? (std::uint64_t{1} << (5u + index)) : std::uint64_t{0};
}
} // namespace ComponentTag

[[nodiscard]] constexpr bool maskContainsAll(ArchetypeMask mask, ArchetypeMask required) noexcept {
    return (mask & required) == required;
}

[[nodiscard]] constexpr bool maskContainsAny(ArchetypeMask mask, ArchetypeMask anyOf) noexcept {
    return (mask & anyOf) != 0;
}

/// AoS transfer record used when inserting, reading, writing, or migrating entities across archetypes.
struct KinematicData {
    float posX = 0.0f;
    float posY = 0.0f;
    float posZ = 0.0f;
    float velX = 0.0f;
    float velY = 0.0f;
    float velZ = 0.0f;
    float accX = 0.0f;
    float accY = 0.0f;
    float accZ = 0.0f;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    float scaleZ = 1.0f;
    float halfExtX = 0.0f;
    float halfExtY = 0.0f;
    float halfExtZ = 0.0f;
};

/// Single-archetype Structure-of-Arrays (SoA) table with 64-byte aligned columns and O(1) entity indexing.
/// Only component columns whose bits are set in `mask()` are populated.
class KinematicSoaTable {
public:
    explicit KinematicSoaTable(ArchetypeMask mask = ComponentTag::AllKinematic);

    [[nodiscard]] ArchetypeMask mask() const noexcept { return mask_; }
    [[nodiscard]] bool hasComponents(ArchetypeMask required) const noexcept {
        return maskContainsAll(mask_, required);
    }

    [[nodiscard]] std::size_t size() const noexcept { return entityIds_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return entityIds_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return entityIds_.empty(); }

    void reserve(std::size_t newCapacity);
    void clear() noexcept;
    void shrinkToFit();

    /// Adds `entityId` with values from `data` for active components. Returns false if already present.
    bool addEntity(EntityId entityId, const KinematicData& data = {});

    /// O(1) swap-and-pop removal by `entityId`. Returns false if not found.
    bool removeEntity(EntityId entityId);

    [[nodiscard]] bool contains(EntityId entityId) const noexcept;
    [[nodiscard]] std::optional<std::size_t> indexOf(EntityId entityId) const noexcept;

    [[nodiscard]] std::optional<KinematicData> readEntity(EntityId entityId) const noexcept;
    [[nodiscard]] KinematicData readAt(std::size_t index) const;

    bool writeEntity(EntityId entityId, const KinematicData& data) noexcept;
    void writeAt(std::size_t index, const KinematicData& data);

    /// Semi-implicit Euler integration over active columns:
    /// - if Velocity & Acceleration: `vel += acc * dt`
    /// - if Position & Velocity:     `pos += vel * dt`
    void integrateKinematics(float dt) noexcept;

    /// Computes world-space AABBs using active Position, Bounds (`halfExt`), and Scale:
    /// `ext = halfExt * abs(scale)`, `min = pos - ext`, `max = pos + ext`.
    /// Absent Position defaults to 0; absent Scale defaults to 1; absent Bounds defaults to 0.
    void computeWorldAabbs(AlignedSoaColumn<float>& minX,
                           AlignedSoaColumn<float>& minY,
                           AlignedSoaColumn<float>& minZ,
                           AlignedSoaColumn<float>& maxX,
                           AlignedSoaColumn<float>& maxY,
                           AlignedSoaColumn<float>& maxZ) const;

    /// Scales and translates active Position columns in-place (`pos' = pos * scale + translate`).
    void batchScaleTranslate(float scaleX,
                             float scaleY,
                             float scaleZ,
                             float translateX,
                             float translateY,
                             float translateZ) noexcept;

    [[nodiscard]] const AlignedSoaColumn<EntityId>& entityIds() const noexcept { return entityIds_; }
    [[nodiscard]] const std::unordered_map<EntityId, std::size_t>& entityIndexMap() const noexcept {
        return entityIndexMap_;
    }

    [[nodiscard]] AlignedSoaColumn<float>& posX() noexcept { return posX_; }
    [[nodiscard]] AlignedSoaColumn<float>& posY() noexcept { return posY_; }
    [[nodiscard]] AlignedSoaColumn<float>& posZ() noexcept { return posZ_; }
    [[nodiscard]] const AlignedSoaColumn<float>& posX() const noexcept { return posX_; }
    [[nodiscard]] const AlignedSoaColumn<float>& posY() const noexcept { return posY_; }
    [[nodiscard]] const AlignedSoaColumn<float>& posZ() const noexcept { return posZ_; }

    [[nodiscard]] AlignedSoaColumn<float>& velX() noexcept { return velX_; }
    [[nodiscard]] AlignedSoaColumn<float>& velY() noexcept { return velY_; }
    [[nodiscard]] AlignedSoaColumn<float>& velZ() noexcept { return velZ_; }
    [[nodiscard]] const AlignedSoaColumn<float>& velX() const noexcept { return velX_; }
    [[nodiscard]] const AlignedSoaColumn<float>& velY() const noexcept { return velY_; }
    [[nodiscard]] const AlignedSoaColumn<float>& velZ() const noexcept { return velZ_; }

    [[nodiscard]] AlignedSoaColumn<float>& accX() noexcept { return accX_; }
    [[nodiscard]] AlignedSoaColumn<float>& accY() noexcept { return accY_; }
    [[nodiscard]] AlignedSoaColumn<float>& accZ() noexcept { return accZ_; }
    [[nodiscard]] const AlignedSoaColumn<float>& accX() const noexcept { return accX_; }
    [[nodiscard]] const AlignedSoaColumn<float>& accY() const noexcept { return accY_; }
    [[nodiscard]] const AlignedSoaColumn<float>& accZ() const noexcept { return accZ_; }

    [[nodiscard]] AlignedSoaColumn<float>& scaleX() noexcept { return scaleX_; }
    [[nodiscard]] AlignedSoaColumn<float>& scaleY() noexcept { return scaleY_; }
    [[nodiscard]] AlignedSoaColumn<float>& scaleZ() noexcept { return scaleZ_; }
    [[nodiscard]] const AlignedSoaColumn<float>& scaleX() const noexcept { return scaleX_; }
    [[nodiscard]] const AlignedSoaColumn<float>& scaleY() const noexcept { return scaleY_; }
    [[nodiscard]] const AlignedSoaColumn<float>& scaleZ() const noexcept { return scaleZ_; }

    [[nodiscard]] AlignedSoaColumn<float>& halfExtX() noexcept { return halfExtX_; }
    [[nodiscard]] AlignedSoaColumn<float>& halfExtY() noexcept { return halfExtY_; }
    [[nodiscard]] AlignedSoaColumn<float>& halfExtZ() noexcept { return halfExtZ_; }
    [[nodiscard]] const AlignedSoaColumn<float>& halfExtX() const noexcept { return halfExtX_; }
    [[nodiscard]] const AlignedSoaColumn<float>& halfExtY() const noexcept { return halfExtY_; }
    [[nodiscard]] const AlignedSoaColumn<float>& halfExtZ() const noexcept { return halfExtZ_; }

private:
    [[nodiscard]] KinematicData readAtUnchecked(std::size_t index) const noexcept;
    void writeAtUnchecked(std::size_t index, const KinematicData& data) noexcept;
    void swapRemoveRow(std::size_t index);

    ArchetypeMask mask_ = ComponentTag::AllKinematic;
    AlignedSoaColumn<EntityId> entityIds_;
    AlignedSoaColumn<float> posX_, posY_, posZ_;
    AlignedSoaColumn<float> velX_, velY_, velZ_;
    AlignedSoaColumn<float> accX_, accY_, accZ_;
    AlignedSoaColumn<float> scaleX_, scaleY_, scaleZ_;
    AlignedSoaColumn<float> halfExtX_, halfExtY_, halfExtZ_;
    std::unordered_map<EntityId, std::size_t> entityIndexMap_;
};

using ArchetypeTable = KinematicSoaTable;
using ArchetypeChunk = KinematicSoaTable;

/// Registry partitioning entities into per-archetype SoA tables with O(1) lookup and migration.
class ArchetypeRegistry {
public:
    ArchetypeRegistry() = default;
    ~ArchetypeRegistry() = default;

    ArchetypeRegistry(const ArchetypeRegistry&) = delete;
    ArchetypeRegistry& operator=(const ArchetypeRegistry&) = delete;
    ArchetypeRegistry(ArchetypeRegistry&&) noexcept = default;
    ArchetypeRegistry& operator=(ArchetypeRegistry&&) noexcept = default;

    [[nodiscard]] KinematicSoaTable& getOrCreateTable(ArchetypeMask mask);
    [[nodiscard]] KinematicSoaTable* findTable(ArchetypeMask mask) noexcept;
    [[nodiscard]] const KinematicSoaTable* findTable(ArchetypeMask mask) const noexcept;

    /// Adds `entityId` into the table matching `mask`. Returns false if `entityId` already exists.
    bool addEntity(EntityId entityId, ArchetypeMask mask, const KinematicData& data = {});

    /// Removes `entityId` in O(1) via swap-and-pop. Returns false if not found.
    bool removeEntity(EntityId entityId);

    /// Moves `entityId` to `newMask`, preserving all shared component fields and initializing
    /// newly added components to `KinematicData` defaults (`scale = 1`, others `0`).
    bool migrateEntity(EntityId entityId, ArchetypeMask newMask);

    /// Adds component bits to `entityId`, initializing only newly added components from `newValues`.
    bool addComponents(EntityId entityId, ArchetypeMask addedMask, const KinematicData& newValues = {});

    /// Removes component bits from `entityId`, preserving all remaining components.
    bool removeComponents(EntityId entityId, ArchetypeMask removedMask);

    [[nodiscard]] bool contains(EntityId entityId) const noexcept;
    [[nodiscard]] std::optional<ArchetypeMask> archetypeOf(EntityId entityId) const noexcept;
    [[nodiscard]] KinematicSoaTable* findEntityTable(EntityId entityId) noexcept;
    [[nodiscard]] const KinematicSoaTable* findEntityTable(EntityId entityId) const noexcept;

    [[nodiscard]] std::optional<KinematicData> readEntity(EntityId entityId) const noexcept;
    bool writeEntity(EntityId entityId, const KinematicData& data) noexcept;

    [[nodiscard]] std::size_t entityCount() const noexcept { return entityTableMap_.size(); }
    [[nodiscard]] std::size_t archetypeCount() const noexcept { return tables_.size(); }

    void clear() noexcept;
    void pruneEmptyTables();

    /// Invokes `fn(KinematicSoaTable&)` on every non-empty table matching `requiredMask`.
    template <class Fn>
    void forEachArchetype(ArchetypeMask requiredMask, Fn&& fn) {
        for (std::size_t i = 0; i < tables_.size(); ++i) {
            KinematicSoaTable* table = tables_[i].get();
            if (table != nullptr && !table->empty() && table->hasComponents(requiredMask)) {
                fn(*table);
            }
        }
    }

    template <class Fn>
    void forEachArchetype(ArchetypeMask requiredMask, Fn&& fn) const {
        for (std::size_t i = 0; i < tables_.size(); ++i) {
            const KinematicSoaTable* table = tables_[i].get();
            if (table != nullptr && !table->empty() && table->hasComponents(requiredMask)) {
                fn(*table);
            }
        }
    }

    template <class Fn>
    void forEachArchetype(Fn&& fn) {
        forEachArchetype(ComponentTag::None, std::forward<Fn>(fn));
    }

    template <class Fn>
    void forEachArchetype(Fn&& fn) const {
        forEachArchetype(ComponentTag::None, std::forward<Fn>(fn));
    }

    void integrateKinematics(float dt, ArchetypeMask requiredMask = ComponentTag::None);

    void batchScaleTranslate(float scaleX,
                             float scaleY,
                             float scaleZ,
                             float translateX,
                             float translateY,
                             float translateZ,
                             ArchetypeMask requiredMask = ComponentTag::Position);

private:
    std::vector<std::unique_ptr<KinematicSoaTable>> tables_;
    std::unordered_map<ArchetypeMask, std::size_t> maskToTableIndex_;
    std::unordered_map<EntityId, KinematicSoaTable*> entityTableMap_;
};

/// Standalone SIMD-friendly batch kernel: `out[i] = in[i] * scale + translate` across 3D SoA spans.
/// All input/output spans must have identical length. Supports in-place aliasing (`in == out`).
void batchScaleTranslate(std::span<const float> inX,
                         std::span<const float> inY,
                         std::span<const float> inZ,
                         std::span<float> outX,
                         std::span<float> outY,
                         std::span<float> outZ,
                         float scaleX,
                         float scaleY,
                         float scaleZ,
                         float translateX,
                         float translateY,
                         float translateZ);

/// Standalone in-place SIMD batch kernel over 64B-aligned SoA columns (`pos' = pos * scale + translate`).
void batchScaleTranslate(AlignedSoaColumn<float>& x,
                         AlignedSoaColumn<float>& y,
                         AlignedSoaColumn<float>& z,
                         float scaleX,
                         float scaleY,
                         float scaleZ,
                         float translateX,
                         float translateY,
                         float translateZ);

template <class Fn>
void forEachArchetype(ArchetypeRegistry& registry, ArchetypeMask requiredMask, Fn&& fn) {
    registry.forEachArchetype(requiredMask, std::forward<Fn>(fn));
}

template <class Fn>
void forEachArchetype(const ArchetypeRegistry& registry, ArchetypeMask requiredMask, Fn&& fn) {
    registry.forEachArchetype(requiredMask, std::forward<Fn>(fn));
}

} // namespace lumora
