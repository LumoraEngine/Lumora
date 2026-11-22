#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324) // Structure was padded due to cache-line alignas(64) specifier.
#endif

namespace lumora {

namespace detail {

inline constexpr std::size_t kCacheLineBytes = 64;

template <typename Element, std::size_t StaticCapacity>
class QueueStorage {
public:
    explicit QueueStorage(std::size_t /*capacity*/) noexcept {}

    [[nodiscard]] Element* data() noexcept { return elements_; }
    [[nodiscard]] const Element* data() const noexcept { return elements_; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return StaticCapacity; }

private:
    Element elements_[StaticCapacity]{};
};

template <typename Element>
class QueueStorage<Element, 0> {
public:
    explicit QueueStorage(std::size_t capacity)
        : capacity_(std::bit_ceil(std::max<std::size_t>(capacity, 2)))
        , elements_(allocate(capacity_)) {}

    ~QueueStorage() { deallocate(elements_, capacity_); }

    QueueStorage(const QueueStorage&) = delete;
    QueueStorage& operator=(const QueueStorage&) = delete;
    QueueStorage(QueueStorage&&) = delete;
    QueueStorage& operator=(QueueStorage&&) = delete;

    [[nodiscard]] Element* data() noexcept { return elements_; }
    [[nodiscard]] const Element* data() const noexcept { return elements_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

private:
    static Element* allocate(std::size_t count) {
        auto* raw = static_cast<Element*>(
            ::operator new[](count * sizeof(Element), std::align_val_t{alignof(Element)}));
        for (std::size_t i = 0; i < count; ++i) {
            std::construct_at(raw + i);
        }
        return raw;
    }

    static void deallocate(Element* ptr, std::size_t count) noexcept {
        if (!ptr) {
            return;
        }
        for (std::size_t i = count; i > 0; --i) {
            std::destroy_at(ptr + (i - 1));
        }
        ::operator delete[](ptr, std::align_val_t{alignof(Element)});
    }

    std::size_t capacity_{0};
    Element* elements_{nullptr};
};

} // namespace detail

/// Wait-free bounded single-producer single-consumer (SPSC) ring buffer.
/// Capacity is always a power of two (>= 2).
/// Supports either a compile-time capacity `SpscRingBuffer<T, 1024>` or a
/// runtime capacity `SpscRingBuffer<T>(1024)` when `StaticCapacity == 0`.
template <typename T, std::size_t StaticCapacity = 0>
class SpscRingBuffer {
    static_assert(StaticCapacity == 0 || (StaticCapacity >= 2 && std::has_single_bit(StaticCapacity)),
                  "SpscRingBuffer static capacity must be a power of two >= 2");

    struct Slot {
        alignas(T) std::byte storage[sizeof(T)];
    };

public:
    static constexpr std::size_t kDefaultCapacity = 1024;

    SpscRingBuffer() requires(StaticCapacity != 0)
        : storage_(StaticCapacity)
        , capacity_(StaticCapacity)
        , mask_(StaticCapacity - 1) {}

    explicit SpscRingBuffer(std::size_t capacity = kDefaultCapacity) requires(StaticCapacity == 0)
        : storage_(capacity)
        , capacity_(storage_.capacity())
        , mask_(capacity_ - 1) {}

    ~SpscRingBuffer() {
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_relaxed);
        while (tail != head) {
            std::destroy_at(slotPtr(tail & mask_));
            ++tail;
        }
    }

    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;
    SpscRingBuffer(SpscRingBuffer&&) = delete;
    SpscRingBuffer& operator=(SpscRingBuffer&&) = delete;

    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    bool tryEmplace(Args&&... args) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head - cachedTail_ >= capacity_) {
            cachedTail_ = tail_.load(std::memory_order_acquire);
            if (head - cachedTail_ >= capacity_) {
                return false;
            }
        }
        std::construct_at(slotPtr(head & mask_), std::forward<Args>(args)...);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool tryPush(const T& value) requires std::is_copy_constructible_v<T> {
        return tryEmplace(value);
    }

    bool tryPush(T&& value) requires std::is_move_constructible_v<T> {
        return tryEmplace(std::move(value));
    }

    bool tryPop(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (cachedHead_ == tail) {
            cachedHead_ = head_.load(std::memory_order_acquire);
            if (cachedHead_ == tail) {
                return false;
            }
        }
        T* ptr = slotPtr(tail & mask_);
        out = std::move(*ptr);
        std::destroy_at(ptr);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t sizeApprox() const noexcept {
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (head <= tail) {
            return 0;
        }
        return std::min<std::size_t>(head - tail, capacity_);
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] bool empty() const noexcept {
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t head = head_.load(std::memory_order_acquire);
        return head == tail;
    }

private:
    [[nodiscard]] T* slotPtr(std::size_t index) noexcept {
        return std::launder(static_cast<T*>(static_cast<void*>(storage_.data()[index].storage)));
    }

    detail::QueueStorage<Slot, StaticCapacity> storage_;
    std::size_t capacity_{0};
    std::size_t mask_{0};

    alignas(detail::kCacheLineBytes) std::atomic<std::size_t> head_{0};
    std::size_t cachedTail_{0};

    alignas(detail::kCacheLineBytes) std::atomic<std::size_t> tail_{0};
    std::size_t cachedHead_{0};
};

/// Vyukov bounded lock-free multi-producer multi-consumer (MPMC) queue.
/// Uses per-cell atomic sequence numbers with cache-line padding to eliminate false sharing.
/// Capacity is always a power of two (>= 2).
template <typename T, std::size_t StaticCapacity = 0>
class MpmcBoundedQueue {
    static_assert(StaticCapacity == 0 || (StaticCapacity >= 2 && std::has_single_bit(StaticCapacity)),
                  "MpmcBoundedQueue static capacity must be a power of two >= 2");

    struct alignas(std::max<std::size_t>(detail::kCacheLineBytes, alignof(T))) Cell {
        std::atomic<std::size_t> sequence{0};
        alignas(T) std::byte storage[sizeof(T)]{};
    };

    using SignedIndex = std::make_signed_t<std::size_t>;

public:
    static constexpr std::size_t kDefaultCapacity = 1024;

    MpmcBoundedQueue() requires(StaticCapacity != 0)
        : storage_(StaticCapacity)
        , capacity_(StaticCapacity)
        , mask_(StaticCapacity - 1) {
        initSequences();
    }

    explicit MpmcBoundedQueue(std::size_t capacity = kDefaultCapacity) requires(StaticCapacity == 0)
        : storage_(capacity)
        , capacity_(storage_.capacity())
        , mask_(capacity_ - 1) {
        initSequences();
    }

    ~MpmcBoundedQueue() {
        std::size_t pos = dequeuePos_.load(std::memory_order_relaxed);
        const std::size_t end = enqueuePos_.load(std::memory_order_relaxed);
        Cell* cells = storage_.data();
        while (pos != end) {
            Cell& cell = cells[pos & mask_];
            if (cell.sequence.load(std::memory_order_relaxed) == pos + 1) {
                std::destroy_at(cellPtr(cell));
            }
            ++pos;
        }
    }

    MpmcBoundedQueue(const MpmcBoundedQueue&) = delete;
    MpmcBoundedQueue& operator=(const MpmcBoundedQueue&) = delete;
    MpmcBoundedQueue(MpmcBoundedQueue&&) = delete;
    MpmcBoundedQueue& operator=(MpmcBoundedQueue&&) = delete;

    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    bool tryEmplace(Args&&... args) {
        Cell* cells = storage_.data();
        Cell* cell = nullptr;
        std::size_t pos = enqueuePos_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &cells[pos & mask_];
            const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<SignedIndex>(seq - pos);
            if (diff == 0) {
                if (enqueuePos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueuePos_.load(std::memory_order_relaxed);
            }
        }

        std::construct_at(cellPtr(*cell), std::forward<Args>(args)...);
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool tryPush(const T& value) requires std::is_copy_constructible_v<T> {
        return tryEmplace(value);
    }

    bool tryPush(T&& value) requires std::is_move_constructible_v<T> {
        return tryEmplace(std::move(value));
    }

    bool tryPop(T& out) {
        Cell* cells = storage_.data();
        Cell* cell = nullptr;
        std::size_t pos = dequeuePos_.load(std::memory_order_relaxed);

        for (;;) {
            cell = &cells[pos & mask_];
            const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<SignedIndex>(seq - (pos + 1));
            if (diff == 0) {
                if (dequeuePos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeuePos_.load(std::memory_order_relaxed);
            }
        }

        T* ptr = cellPtr(*cell);
        out = std::move(*ptr);
        std::destroy_at(ptr);
        cell->sequence.store(pos + mask_ + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::size_t sizeApprox() const noexcept {
        const std::size_t deq = dequeuePos_.load(std::memory_order_acquire);
        const std::size_t enq = enqueuePos_.load(std::memory_order_acquire);
        if (enq <= deq) {
            return 0;
        }
        return std::min<std::size_t>(enq - deq, capacity_);
    }

    [[nodiscard]] bool emptyApprox() const noexcept {
        const std::size_t deq = dequeuePos_.load(std::memory_order_acquire);
        const std::size_t enq = enqueuePos_.load(std::memory_order_acquire);
        return enq == deq;
    }

private:
    void initSequences() noexcept {
        Cell* cells = storage_.data();
        for (std::size_t i = 0; i < capacity_; ++i) {
            cells[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] static T* cellPtr(Cell& cell) noexcept {
        return std::launder(static_cast<T*>(static_cast<void*>(cell.storage)));
    }

    detail::QueueStorage<Cell, StaticCapacity> storage_;
    std::size_t capacity_{0};
    std::size_t mask_{0};

    alignas(detail::kCacheLineBytes) std::atomic<std::size_t> enqueuePos_{0};
    alignas(detail::kCacheLineBytes) std::atomic<std::size_t> dequeuePos_{0};
};

} // namespace lumora

#ifdef _MSC_VER
#pragma warning(pop)
#endif
