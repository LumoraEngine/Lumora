#include "core/ArchetypeStore.hpp"

#include <array>
#include <utility>

#if defined(__SSE4_2__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace lumora {
namespace {

[[nodiscard]] inline bool isSimd16Aligned(const void* ptr) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 15u) == 0u;
}

template <class Table, class Fn>
void visitActiveFloatColumns(Table& table, Fn&& fn) {
    if (table.hasComponents(ComponentTag::Position)) {
        fn(table.posX());
        fn(table.posY());
        fn(table.posZ());
    }
    if (table.hasComponents(ComponentTag::Velocity)) {
        fn(table.velX());
        fn(table.velY());
        fn(table.velZ());
    }
    if (table.hasComponents(ComponentTag::Acceleration)) {
        fn(table.accX());
        fn(table.accY());
        fn(table.accZ());
    }
    if (table.hasComponents(ComponentTag::Scale)) {
        fn(table.scaleX());
        fn(table.scaleY());
        fn(table.scaleZ());
    }
    if (table.hasComponents(ComponentTag::Bounds)) {
        fn(table.halfExtX());
        fn(table.halfExtY());
        fn(table.halfExtZ());
    }
}

void integratePosVelAccAxis(std::size_t count,
                            float* LUMORA_SOA_RESTRICT pos,
                            float* LUMORA_SOA_RESTRICT vel,
                            const float* LUMORA_SOA_RESTRICT acc,
                            float dt) noexcept {
    if (count == 0) {
        return;
    }
    pos = std::assume_aligned<kSoaCacheLineBytes>(pos);
    vel = std::assume_aligned<kSoaCacheLineBytes>(vel);
    acc = std::assume_aligned<kSoaCacheLineBytes>(acc);

    std::size_t i = 0;
#if defined(__SSE4_2__) || defined(_M_X64)
    const __m128 vdt = _mm_set1_ps(dt);
    for (; i + 8 <= count; i += 8) {
        __m128 v0 = _mm_load_ps(vel + i);
        __m128 v1 = _mm_load_ps(vel + i + 4);
        const __m128 a0 = _mm_load_ps(acc + i);
        const __m128 a1 = _mm_load_ps(acc + i + 4);
        v0 = _mm_add_ps(v0, _mm_mul_ps(a0, vdt));
        v1 = _mm_add_ps(v1, _mm_mul_ps(a1, vdt));
        _mm_store_ps(vel + i, v0);
        _mm_store_ps(vel + i + 4, v1);

        __m128 p0 = _mm_load_ps(pos + i);
        __m128 p1 = _mm_load_ps(pos + i + 4);
        p0 = _mm_add_ps(p0, _mm_mul_ps(v0, vdt));
        p1 = _mm_add_ps(p1, _mm_mul_ps(v1, vdt));
        _mm_store_ps(pos + i, p0);
        _mm_store_ps(pos + i + 4, p1);
    }
    for (; i + 4 <= count; i += 4) {
        __m128 v0 = _mm_load_ps(vel + i);
        const __m128 a0 = _mm_load_ps(acc + i);
        v0 = _mm_add_ps(v0, _mm_mul_ps(a0, vdt));
        _mm_store_ps(vel + i, v0);

        __m128 p0 = _mm_load_ps(pos + i);
        p0 = _mm_add_ps(p0, _mm_mul_ps(v0, vdt));
        _mm_store_ps(pos + i, p0);
    }
#endif
    for (; i < count; ++i) {
        const float v = vel[i] + acc[i] * dt;
        vel[i] = v;
        pos[i] = pos[i] + v * dt;
    }
}

void integrateLinearAxis(std::size_t count,
                         float* LUMORA_SOA_RESTRICT dst,
                         const float* LUMORA_SOA_RESTRICT src,
                         float dt) noexcept {
    if (count == 0) {
        return;
    }
    dst = std::assume_aligned<kSoaCacheLineBytes>(dst);
    src = std::assume_aligned<kSoaCacheLineBytes>(src);

    std::size_t i = 0;
#if defined(__SSE4_2__) || defined(_M_X64)
    const __m128 vdt = _mm_set1_ps(dt);
    for (; i + 8 <= count; i += 8) {
        __m128 d0 = _mm_load_ps(dst + i);
        __m128 d1 = _mm_load_ps(dst + i + 4);
        const __m128 s0 = _mm_load_ps(src + i);
        const __m128 s1 = _mm_load_ps(src + i + 4);
        d0 = _mm_add_ps(d0, _mm_mul_ps(s0, vdt));
        d1 = _mm_add_ps(d1, _mm_mul_ps(s1, vdt));
        _mm_store_ps(dst + i, d0);
        _mm_store_ps(dst + i + 4, d1);
    }
    for (; i + 4 <= count; i += 4) {
        __m128 d0 = _mm_load_ps(dst + i);
        const __m128 s0 = _mm_load_ps(src + i);
        d0 = _mm_add_ps(d0, _mm_mul_ps(s0, vdt));
        _mm_store_ps(dst + i, d0);
    }
#endif
    for (; i < count; ++i) {
        dst[i] = dst[i] + src[i] * dt;
    }
}

void computeAabbPosBoundsScaleAxis(std::size_t count,
                                  const float* LUMORA_SOA_RESTRICT pos,
                                  const float* LUMORA_SOA_RESTRICT halfExt,
                                  const float* LUMORA_SOA_RESTRICT scale,
                                  float* LUMORA_SOA_RESTRICT outMin,
                                  float* LUMORA_SOA_RESTRICT outMax) noexcept {
    pos = std::assume_aligned<kSoaCacheLineBytes>(pos);
    halfExt = std::assume_aligned<kSoaCacheLineBytes>(halfExt);
    scale = std::assume_aligned<kSoaCacheLineBytes>(scale);
    outMin = std::assume_aligned<kSoaCacheLineBytes>(outMin);
    outMax = std::assume_aligned<kSoaCacheLineBytes>(outMax);

    std::size_t i = 0;
#if defined(__SSE4_2__) || defined(_M_X64)
    const __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x7FFFFFFFu)));
    for (; i + 8 <= count; i += 8) {
        const __m128 absS0 = _mm_and_ps(_mm_load_ps(scale + i), absMask);
        const __m128 absS1 = _mm_and_ps(_mm_load_ps(scale + i + 4), absMask);
        const __m128 ext0 = _mm_mul_ps(_mm_load_ps(halfExt + i), absS0);
        const __m128 ext1 = _mm_mul_ps(_mm_load_ps(halfExt + i + 4), absS1);
        const __m128 p0 = _mm_load_ps(pos + i);
        const __m128 p1 = _mm_load_ps(pos + i + 4);
        _mm_store_ps(outMin + i, _mm_sub_ps(p0, ext0));
        _mm_store_ps(outMin + i + 4, _mm_sub_ps(p1, ext1));
        _mm_store_ps(outMax + i, _mm_add_ps(p0, ext0));
        _mm_store_ps(outMax + i + 4, _mm_add_ps(p1, ext1));
    }
    for (; i + 4 <= count; i += 4) {
        const __m128 absS0 = _mm_and_ps(_mm_load_ps(scale + i), absMask);
        const __m128 ext0 = _mm_mul_ps(_mm_load_ps(halfExt + i), absS0);
        const __m128 p0 = _mm_load_ps(pos + i);
        _mm_store_ps(outMin + i, _mm_sub_ps(p0, ext0));
        _mm_store_ps(outMax + i, _mm_add_ps(p0, ext0));
    }
#endif
    for (; i < count; ++i) {
        const float ext = halfExt[i] * std::abs(scale[i]);
        outMin[i] = pos[i] - ext;
        outMax[i] = pos[i] + ext;
    }
}

void computeAabbPosBoundsAxis(std::size_t count,
                             const float* LUMORA_SOA_RESTRICT pos,
                             const float* LUMORA_SOA_RESTRICT halfExt,
                             float* LUMORA_SOA_RESTRICT outMin,
                             float* LUMORA_SOA_RESTRICT outMax) noexcept {
    pos = std::assume_aligned<kSoaCacheLineBytes>(pos);
    halfExt = std::assume_aligned<kSoaCacheLineBytes>(halfExt);
    outMin = std::assume_aligned<kSoaCacheLineBytes>(outMin);
    outMax = std::assume_aligned<kSoaCacheLineBytes>(outMax);

    std::size_t i = 0;
#if defined(__SSE4_2__) || defined(_M_X64)
    for (; i + 8 <= count; i += 8) {
        const __m128 ext0 = _mm_load_ps(halfExt + i);
        const __m128 ext1 = _mm_load_ps(halfExt + i + 4);
        const __m128 p0 = _mm_load_ps(pos + i);
        const __m128 p1 = _mm_load_ps(pos + i + 4);
        _mm_store_ps(outMin + i, _mm_sub_ps(p0, ext0));
        _mm_store_ps(outMin + i + 4, _mm_sub_ps(p1, ext1));
        _mm_store_ps(outMax + i, _mm_add_ps(p0, ext0));
        _mm_store_ps(outMax + i + 4, _mm_add_ps(p1, ext1));
    }
    for (; i + 4 <= count; i += 4) {
        const __m128 ext0 = _mm_load_ps(halfExt + i);
        const __m128 p0 = _mm_load_ps(pos + i);
        _mm_store_ps(outMin + i, _mm_sub_ps(p0, ext0));
        _mm_store_ps(outMax + i, _mm_add_ps(p0, ext0));
    }
#endif
    for (; i < count; ++i) {
        outMin[i] = pos[i] - halfExt[i];
        outMax[i] = pos[i] + halfExt[i];
    }
}

void computeAabbAxisDispatch(std::size_t count,
                             bool hasPos,
                             bool hasBounds,
                             bool hasScale,
                             const AlignedSoaColumn<float>& posCol,
                             const AlignedSoaColumn<float>& halfExtCol,
                             const AlignedSoaColumn<float>& scaleCol,
                             AlignedSoaColumn<float>& outMinCol,
                             AlignedSoaColumn<float>& outMaxCol) {
    outMinCol.resize(count);
    outMaxCol.resize(count);
    if (count == 0) {
        return;
    }

    float* LUMORA_SOA_RESTRICT outMin = outMinCol.assumeAlignedData();
    float* LUMORA_SOA_RESTRICT outMax = outMaxCol.assumeAlignedData();

    if (hasPos && hasBounds && hasScale) {
        computeAabbPosBoundsScaleAxis(count, posCol.assumeAlignedData(), halfExtCol.assumeAlignedData(),
                                      scaleCol.assumeAlignedData(), outMin, outMax);
        return;
    }
    if (hasPos && hasBounds) {
        computeAabbPosBoundsAxis(count, posCol.assumeAlignedData(), halfExtCol.assumeAlignedData(), outMin, outMax);
        return;
    }
    if (hasPos) {
        const float* LUMORA_SOA_RESTRICT pos = posCol.assumeAlignedData();
        std::copy_n(pos, count, outMin);
        std::copy_n(pos, count, outMax);
        return;
    }
    if (hasBounds) {
        const float* LUMORA_SOA_RESTRICT halfExt = halfExtCol.assumeAlignedData();
        if (hasScale) {
            const float* LUMORA_SOA_RESTRICT scale = scaleCol.assumeAlignedData();
            for (std::size_t i = 0; i < count; ++i) {
                const float ext = halfExt[i] * std::abs(scale[i]);
                outMin[i] = -ext;
                outMax[i] = ext;
            }
        } else {
            for (std::size_t i = 0; i < count; ++i) {
                const float ext = halfExt[i];
                outMin[i] = -ext;
                outMax[i] = ext;
            }
        }
        return;
    }
    std::fill_n(outMin, count, 0.0f);
    std::fill_n(outMax, count, 0.0f);
}

template <std::size_t Align>
void scaleTranslateInPlaceAxis(std::size_t count,
                               float* LUMORA_SOA_RESTRICT data,
                               float scale,
                               float translate) noexcept {
    if (count == 0) {
        return;
    }
    if constexpr (Align >= 16) {
        data = std::assume_aligned<Align>(data);
    }

    std::size_t i = 0;
#if defined(__SSE4_2__) || defined(_M_X64)
    const __m128 vs = _mm_set1_ps(scale);
    const __m128 vt = _mm_set1_ps(translate);
    auto load4 = [](const float* p) noexcept -> __m128 {
        if constexpr (Align >= 16) {
            return _mm_load_ps(p);
        } else {
            return _mm_loadu_ps(p);
        }
    };
    auto store4 = [](float* p, __m128 v) noexcept {
        if constexpr (Align >= 16) {
            _mm_store_ps(p, v);
        } else {
            _mm_storeu_ps(p, v);
        }
    };
    for (; i + 8 <= count; i += 8) {
        const __m128 v0 = _mm_add_ps(_mm_mul_ps(load4(data + i), vs), vt);
        const __m128 v1 = _mm_add_ps(_mm_mul_ps(load4(data + i + 4), vs), vt);
        store4(data + i, v0);
        store4(data + i + 4, v1);
    }
    for (; i + 4 <= count; i += 4) {
        const __m128 v0 = _mm_add_ps(_mm_mul_ps(load4(data + i), vs), vt);
        store4(data + i, v0);
    }
#endif
    for (; i < count; ++i) {
        data[i] = data[i] * scale + translate;
    }
}

template <std::size_t Align>
void scaleTranslateDisjointAxis(std::size_t count,
                                const float* LUMORA_SOA_RESTRICT in,
                                float* LUMORA_SOA_RESTRICT out,
                                float scale,
                                float translate) noexcept {
    if (count == 0) {
        return;
    }
    if constexpr (Align >= 16) {
        in = std::assume_aligned<Align>(in);
        out = std::assume_aligned<Align>(out);
    }

    std::size_t i = 0;
#if defined(__SSE4_2__) || defined(_M_X64)
    const __m128 vs = _mm_set1_ps(scale);
    const __m128 vt = _mm_set1_ps(translate);
    auto load4 = [](const float* p) noexcept -> __m128 {
        if constexpr (Align >= 16) {
            return _mm_load_ps(p);
        } else {
            return _mm_loadu_ps(p);
        }
    };
    auto store4 = [](float* p, __m128 v) noexcept {
        if constexpr (Align >= 16) {
            _mm_store_ps(p, v);
        } else {
            _mm_storeu_ps(p, v);
        }
    };
    for (; i + 8 <= count; i += 8) {
        const __m128 v0 = _mm_add_ps(_mm_mul_ps(load4(in + i), vs), vt);
        const __m128 v1 = _mm_add_ps(_mm_mul_ps(load4(in + i + 4), vs), vt);
        store4(out + i, v0);
        store4(out + i + 4, v1);
    }
    for (; i + 4 <= count; i += 4) {
        const __m128 v0 = _mm_add_ps(_mm_mul_ps(load4(in + i), vs), vt);
        store4(out + i, v0);
    }
#endif
    for (; i < count; ++i) {
        out[i] = in[i] * scale + translate;
    }
}

void scaleTranslateSpanAxis(std::span<const float> inSpan,
                            std::span<float> outSpan,
                            float scale,
                            float translate) noexcept {
    const std::size_t count = inSpan.size();
    if (count == 0) {
        return;
    }
    const float* inPtr = inSpan.data();
    float* outPtr = outSpan.data();

    if (inPtr == outPtr) {
        if (isCacheLineAligned(outPtr)) {
            scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, outPtr, scale, translate);
        } else if (isSimd16Aligned(outPtr)) {
            scaleTranslateInPlaceAxis<16>(count, outPtr, scale, translate);
        } else {
            scaleTranslateInPlaceAxis<1>(count, outPtr, scale, translate);
        }
        return;
    }

    const std::less<const float*> ptrLess{};
    const bool disjoint = !ptrLess(outPtr, inPtr + count) || !ptrLess(inPtr, outPtr + count);
    if (disjoint) {
        if (isCacheLineAligned(inPtr) && isCacheLineAligned(outPtr)) {
            scaleTranslateDisjointAxis<kSoaCacheLineBytes>(count, inPtr, outPtr, scale, translate);
        } else if (isSimd16Aligned(inPtr) && isSimd16Aligned(outPtr)) {
            scaleTranslateDisjointAxis<16>(count, inPtr, outPtr, scale, translate);
        } else {
            scaleTranslateDisjointAxis<1>(count, inPtr, outPtr, scale, translate);
        }
        return;
    }

    // Partial overlap fallback: preserve unconsumed source elements via direction-aware copy.
    if (ptrLess(outPtr, inPtr)) {
        for (std::size_t i = 0; i < count; ++i) {
            outPtr[i] = inPtr[i] * scale + translate;
        }
    } else {
        for (std::size_t i = count; i-- > 0;) {
            outPtr[i] = inPtr[i] * scale + translate;
        }
    }
}

} // namespace

KinematicSoaTable::KinematicSoaTable(ArchetypeMask mask) : mask_(mask) {}

void KinematicSoaTable::reserve(std::size_t newCapacity) {
    entityIds_.reserve(newCapacity);
    visitActiveFloatColumns(*this, [newCapacity](AlignedSoaColumn<float>& col) { col.reserve(newCapacity); });
}

void KinematicSoaTable::clear() noexcept {
    entityIds_.clear();
    visitActiveFloatColumns(*this, [](AlignedSoaColumn<float>& col) noexcept { col.clear(); });
    entityIndexMap_.clear();
}

void KinematicSoaTable::shrinkToFit() {
    entityIds_.shrinkToFit();
    visitActiveFloatColumns(*this, [](AlignedSoaColumn<float>& col) { col.shrinkToFit(); });
}

bool KinematicSoaTable::addEntity(EntityId entityId, const KinematicData& data) {
    if (entityIndexMap_.contains(entityId)) {
        return false;
    }

    const std::size_t newIndex = entityIds_.size();
    if (newIndex == entityIds_.capacity()) {
        const std::size_t nextCap = entityIds_.capacity() == 0 ? 16 : entityIds_.capacity() * 2;
        reserve(nextCap);
    } else {
        reserve(newIndex + 1);
    }

    entityIndexMap_.emplace(entityId, newIndex);
    entityIds_.pushBack(entityId);

    if (hasComponents(ComponentTag::Position)) {
        posX_.pushBack(data.posX);
        posY_.pushBack(data.posY);
        posZ_.pushBack(data.posZ);
    }
    if (hasComponents(ComponentTag::Velocity)) {
        velX_.pushBack(data.velX);
        velY_.pushBack(data.velY);
        velZ_.pushBack(data.velZ);
    }
    if (hasComponents(ComponentTag::Acceleration)) {
        accX_.pushBack(data.accX);
        accY_.pushBack(data.accY);
        accZ_.pushBack(data.accZ);
    }
    if (hasComponents(ComponentTag::Scale)) {
        scaleX_.pushBack(data.scaleX);
        scaleY_.pushBack(data.scaleY);
        scaleZ_.pushBack(data.scaleZ);
    }
    if (hasComponents(ComponentTag::Bounds)) {
        halfExtX_.pushBack(data.halfExtX);
        halfExtY_.pushBack(data.halfExtY);
        halfExtZ_.pushBack(data.halfExtZ);
    }

    return true;
}

void KinematicSoaTable::swapRemoveRow(std::size_t index) {
    entityIds_.swapRemove(index);
    visitActiveFloatColumns(*this, [index](AlignedSoaColumn<float>& col) { col.swapRemove(index); });
}

bool KinematicSoaTable::removeEntity(EntityId entityId) {
    const auto it = entityIndexMap_.find(entityId);
    if (it == entityIndexMap_.end()) {
        return false;
    }

    const std::size_t index = it->second;
    const std::size_t lastIndex = entityIds_.size() - 1;
    entityIndexMap_.erase(it);

    if (index != lastIndex) {
        const EntityId movedEntity = entityIds_[lastIndex];
        entityIndexMap_.at(movedEntity) = index;
    }

    swapRemoveRow(index);
    return true;
}

bool KinematicSoaTable::contains(EntityId entityId) const noexcept {
    return entityIndexMap_.contains(entityId);
}

std::optional<std::size_t> KinematicSoaTable::indexOf(EntityId entityId) const noexcept {
    const auto it = entityIndexMap_.find(entityId);
    if (it == entityIndexMap_.end()) {
        return std::nullopt;
    }
    return it->second;
}

KinematicData KinematicSoaTable::readAtUnchecked(std::size_t index) const noexcept {
    KinematicData out{};
    if (hasComponents(ComponentTag::Position)) {
        out.posX = posX_[index];
        out.posY = posY_[index];
        out.posZ = posZ_[index];
    }
    if (hasComponents(ComponentTag::Velocity)) {
        out.velX = velX_[index];
        out.velY = velY_[index];
        out.velZ = velZ_[index];
    }
    if (hasComponents(ComponentTag::Acceleration)) {
        out.accX = accX_[index];
        out.accY = accY_[index];
        out.accZ = accZ_[index];
    }
    if (hasComponents(ComponentTag::Scale)) {
        out.scaleX = scaleX_[index];
        out.scaleY = scaleY_[index];
        out.scaleZ = scaleZ_[index];
    }
    if (hasComponents(ComponentTag::Bounds)) {
        out.halfExtX = halfExtX_[index];
        out.halfExtY = halfExtY_[index];
        out.halfExtZ = halfExtZ_[index];
    }
    return out;
}

KinematicData KinematicSoaTable::readAt(std::size_t index) const {
    if (index >= entityIds_.size()) {
        throw std::out_of_range("KinematicSoaTable::readAt: index out of range");
    }
    return readAtUnchecked(index);
}

std::optional<KinematicData> KinematicSoaTable::readEntity(EntityId entityId) const noexcept {
    const auto it = entityIndexMap_.find(entityId);
    if (it == entityIndexMap_.end()) {
        return std::nullopt;
    }
    return readAtUnchecked(it->second);
}

void KinematicSoaTable::writeAtUnchecked(std::size_t index, const KinematicData& data) noexcept {
    if (hasComponents(ComponentTag::Position)) {
        posX_[index] = data.posX;
        posY_[index] = data.posY;
        posZ_[index] = data.posZ;
    }
    if (hasComponents(ComponentTag::Velocity)) {
        velX_[index] = data.velX;
        velY_[index] = data.velY;
        velZ_[index] = data.velZ;
    }
    if (hasComponents(ComponentTag::Acceleration)) {
        accX_[index] = data.accX;
        accY_[index] = data.accY;
        accZ_[index] = data.accZ;
    }
    if (hasComponents(ComponentTag::Scale)) {
        scaleX_[index] = data.scaleX;
        scaleY_[index] = data.scaleY;
        scaleZ_[index] = data.scaleZ;
    }
    if (hasComponents(ComponentTag::Bounds)) {
        halfExtX_[index] = data.halfExtX;
        halfExtY_[index] = data.halfExtY;
        halfExtZ_[index] = data.halfExtZ;
    }
}

void KinematicSoaTable::writeAt(std::size_t index, const KinematicData& data) {
    if (index >= entityIds_.size()) {
        throw std::out_of_range("KinematicSoaTable::writeAt: index out of range");
    }
    writeAtUnchecked(index, data);
}

bool KinematicSoaTable::writeEntity(EntityId entityId, const KinematicData& data) noexcept {
    const auto it = entityIndexMap_.find(entityId);
    if (it == entityIndexMap_.end()) {
        return false;
    }
    writeAtUnchecked(it->second, data);
    return true;
}

void KinematicSoaTable::integrateKinematics(float dt) noexcept {
    const std::size_t count = entityIds_.size();
    if (count == 0) {
        return;
    }

    const bool hasPos = hasComponents(ComponentTag::Position);
    const bool hasVel = hasComponents(ComponentTag::Velocity);
    const bool hasAcc = hasComponents(ComponentTag::Acceleration);
    if (!hasVel) {
        return;
    }

    if (hasPos && hasAcc) {
        integratePosVelAccAxis(count, posX_.assumeAlignedData(), velX_.assumeAlignedData(),
                               accX_.assumeAlignedData(), dt);
        integratePosVelAccAxis(count, posY_.assumeAlignedData(), velY_.assumeAlignedData(),
                               accY_.assumeAlignedData(), dt);
        integratePosVelAccAxis(count, posZ_.assumeAlignedData(), velZ_.assumeAlignedData(),
                               accZ_.assumeAlignedData(), dt);
        return;
    }

    if (hasAcc) {
        integrateLinearAxis(count, velX_.assumeAlignedData(), accX_.assumeAlignedData(), dt);
        integrateLinearAxis(count, velY_.assumeAlignedData(), accY_.assumeAlignedData(), dt);
        integrateLinearAxis(count, velZ_.assumeAlignedData(), accZ_.assumeAlignedData(), dt);
    }

    if (hasPos) {
        integrateLinearAxis(count, posX_.assumeAlignedData(), velX_.assumeAlignedData(), dt);
        integrateLinearAxis(count, posY_.assumeAlignedData(), velY_.assumeAlignedData(), dt);
        integrateLinearAxis(count, posZ_.assumeAlignedData(), velZ_.assumeAlignedData(), dt);
    }
}

void KinematicSoaTable::computeWorldAabbs(AlignedSoaColumn<float>& minX,
                                        AlignedSoaColumn<float>& minY,
                                        AlignedSoaColumn<float>& minZ,
                                        AlignedSoaColumn<float>& maxX,
                                        AlignedSoaColumn<float>& maxY,
                                        AlignedSoaColumn<float>& maxZ) const {
    const std::array<const AlignedSoaColumn<float>*, 6> outCols = {&minX, &minY, &minZ, &maxX, &maxY, &maxZ};
    for (std::size_t i = 0; i < outCols.size(); ++i) {
        for (std::size_t j = i + 1; j < outCols.size(); ++j) {
            if (outCols[i] == outCols[j]) {
                throw std::invalid_argument("KinematicSoaTable::computeWorldAabbs: output columns must not alias");
            }
        }
        const std::array<const AlignedSoaColumn<float>*, 9> inCols = {
            &posX_, &posY_, &posZ_, &halfExtX_, &halfExtY_, &halfExtZ_, &scaleX_, &scaleY_, &scaleZ_};
        for (const AlignedSoaColumn<float>* inCol : inCols) {
            if (outCols[i] == inCol) {
                throw std::invalid_argument(
                    "KinematicSoaTable::computeWorldAabbs: output columns must not alias table columns");
            }
        }
    }

    const std::size_t count = entityIds_.size();
    const bool hasPos = hasComponents(ComponentTag::Position);
    const bool hasBounds = hasComponents(ComponentTag::Bounds);
    const bool hasScale = hasComponents(ComponentTag::Scale);

    computeAabbAxisDispatch(count, hasPos, hasBounds, hasScale, posX_, halfExtX_, scaleX_, minX, maxX);
    computeAabbAxisDispatch(count, hasPos, hasBounds, hasScale, posY_, halfExtY_, scaleY_, minY, maxY);
    computeAabbAxisDispatch(count, hasPos, hasBounds, hasScale, posZ_, halfExtZ_, scaleZ_, minZ, maxZ);
}

void KinematicSoaTable::batchScaleTranslate(float scaleX,
                                            float scaleY,
                                            float scaleZ,
                                            float translateX,
                                            float translateY,
                                            float translateZ) noexcept {
    const std::size_t count = entityIds_.size();
    if (count == 0 || !hasComponents(ComponentTag::Position)) {
        return;
    }

    scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, posX_.assumeAlignedData(), scaleX, translateX);
    scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, posY_.assumeAlignedData(), scaleY, translateY);
    scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, posZ_.assumeAlignedData(), scaleZ, translateZ);
}

KinematicSoaTable& ArchetypeRegistry::getOrCreateTable(ArchetypeMask mask) {
    const auto it = maskToTableIndex_.find(mask);
    if (it != maskToTableIndex_.end()) {
        return *tables_[it->second];
    }

    auto table = std::make_unique<KinematicSoaTable>(mask);
    const std::size_t index = tables_.size();
    maskToTableIndex_.emplace(mask, index);
    tables_.push_back(std::move(table));
    return *tables_.back();
}

KinematicSoaTable* ArchetypeRegistry::findTable(ArchetypeMask mask) noexcept {
    const auto it = maskToTableIndex_.find(mask);
    return it == maskToTableIndex_.end() ? nullptr : tables_[it->second].get();
}

const KinematicSoaTable* ArchetypeRegistry::findTable(ArchetypeMask mask) const noexcept {
    const auto it = maskToTableIndex_.find(mask);
    return it == maskToTableIndex_.end() ? nullptr : tables_[it->second].get();
}

bool ArchetypeRegistry::addEntity(EntityId entityId, ArchetypeMask mask, const KinematicData& data) {
    if (entityTableMap_.contains(entityId)) {
        return false;
    }

    KinematicSoaTable& table = getOrCreateTable(mask);
    if (!table.addEntity(entityId, data)) {
        return false;
    }
    entityTableMap_.emplace(entityId, &table);
    return true;
}

bool ArchetypeRegistry::removeEntity(EntityId entityId) {
    const auto it = entityTableMap_.find(entityId);
    if (it == entityTableMap_.end()) {
        return false;
    }

    KinematicSoaTable* table = it->second;
    entityTableMap_.erase(it);
    return table->removeEntity(entityId);
}

bool ArchetypeRegistry::migrateEntity(EntityId entityId, ArchetypeMask newMask) {
    const auto it = entityTableMap_.find(entityId);
    if (it == entityTableMap_.end()) {
        return false;
    }

    KinematicSoaTable* srcTable = it->second;
    if (srcTable->mask() == newMask) {
        return true;
    }

    const std::optional<std::size_t> row = srcTable->indexOf(entityId);
    if (!row.has_value()) {
        return false;
    }
    const KinematicData snapshot = srcTable->readAt(*row);

    KinematicSoaTable& dstTable = getOrCreateTable(newMask);
    if (!dstTable.addEntity(entityId, snapshot)) {
        return false;
    }

    srcTable->removeEntity(entityId);
    entityTableMap_[entityId] = &dstTable;
    return true;
}

bool ArchetypeRegistry::addComponents(EntityId entityId, ArchetypeMask addedMask, const KinematicData& newValues) {
    const auto it = entityTableMap_.find(entityId);
    if (it == entityTableMap_.end()) {
        return false;
    }

    const ArchetypeMask oldMask = it->second->mask();
    const ArchetypeMask actuallyAdded = addedMask & ~oldMask;
    const ArchetypeMask newMask = oldMask | addedMask;
    if (newMask == oldMask) {
        return true;
    }

    if (!migrateEntity(entityId, newMask)) {
        return false;
    }

    if (actuallyAdded != ComponentTag::None) {
        KinematicSoaTable* dstTable = entityTableMap_.at(entityId);
        const std::optional<std::size_t> row = dstTable->indexOf(entityId);
        if (row.has_value()) {
            KinematicData merged = dstTable->readAt(*row);
            if (maskContainsAll(actuallyAdded, ComponentTag::Position)) {
                merged.posX = newValues.posX;
                merged.posY = newValues.posY;
                merged.posZ = newValues.posZ;
            }
            if (maskContainsAll(actuallyAdded, ComponentTag::Velocity)) {
                merged.velX = newValues.velX;
                merged.velY = newValues.velY;
                merged.velZ = newValues.velZ;
            }
            if (maskContainsAll(actuallyAdded, ComponentTag::Acceleration)) {
                merged.accX = newValues.accX;
                merged.accY = newValues.accY;
                merged.accZ = newValues.accZ;
            }
            if (maskContainsAll(actuallyAdded, ComponentTag::Scale)) {
                merged.scaleX = newValues.scaleX;
                merged.scaleY = newValues.scaleY;
                merged.scaleZ = newValues.scaleZ;
            }
            if (maskContainsAll(actuallyAdded, ComponentTag::Bounds)) {
                merged.halfExtX = newValues.halfExtX;
                merged.halfExtY = newValues.halfExtY;
                merged.halfExtZ = newValues.halfExtZ;
            }
            dstTable->writeAt(*row, merged);
        }
    }

    return true;
}

bool ArchetypeRegistry::removeComponents(EntityId entityId, ArchetypeMask removedMask) {
    const auto it = entityTableMap_.find(entityId);
    if (it == entityTableMap_.end()) {
        return false;
    }
    const ArchetypeMask newMask = it->second->mask() & ~removedMask;
    return migrateEntity(entityId, newMask);
}

bool ArchetypeRegistry::contains(EntityId entityId) const noexcept {
    return entityTableMap_.contains(entityId);
}

std::optional<ArchetypeMask> ArchetypeRegistry::archetypeOf(EntityId entityId) const noexcept {
    const auto it = entityTableMap_.find(entityId);
    if (it == entityTableMap_.end()) {
        return std::nullopt;
    }
    return it->second->mask();
}

KinematicSoaTable* ArchetypeRegistry::findEntityTable(EntityId entityId) noexcept {
    const auto it = entityTableMap_.find(entityId);
    return it == entityTableMap_.end() ? nullptr : it->second;
}

const KinematicSoaTable* ArchetypeRegistry::findEntityTable(EntityId entityId) const noexcept {
    const auto it = entityTableMap_.find(entityId);
    return it == entityTableMap_.end() ? nullptr : it->second;
}

std::optional<KinematicData> ArchetypeRegistry::readEntity(EntityId entityId) const noexcept {
    const KinematicSoaTable* table = findEntityTable(entityId);
    return table == nullptr ? std::nullopt : table->readEntity(entityId);
}

bool ArchetypeRegistry::writeEntity(EntityId entityId, const KinematicData& data) noexcept {
    KinematicSoaTable* table = findEntityTable(entityId);
    return table != nullptr && table->writeEntity(entityId, data);
}

void ArchetypeRegistry::clear() noexcept {
    entityTableMap_.clear();
    maskToTableIndex_.clear();
    tables_.clear();
}

void ArchetypeRegistry::pruneEmptyTables() {
    std::erase_if(tables_, [](const std::unique_ptr<KinematicSoaTable>& table) {
        return table == nullptr || table->empty();
    });
    maskToTableIndex_.clear();
    maskToTableIndex_.reserve(tables_.size());
    for (std::size_t i = 0; i < tables_.size(); ++i) {
        maskToTableIndex_.emplace(tables_[i]->mask(), i);
    }
}

void ArchetypeRegistry::integrateKinematics(float dt, ArchetypeMask requiredMask) {
    forEachArchetype(requiredMask, [dt](KinematicSoaTable& table) { table.integrateKinematics(dt); });
}

void ArchetypeRegistry::batchScaleTranslate(float scaleX,
                                            float scaleY,
                                            float scaleZ,
                                            float translateX,
                                            float translateY,
                                            float translateZ,
                                            ArchetypeMask requiredMask) {
    forEachArchetype(requiredMask, [scaleX, scaleY, scaleZ, translateX, translateY, translateZ](
                                       KinematicSoaTable& table) {
        table.batchScaleTranslate(scaleX, scaleY, scaleZ, translateX, translateY, translateZ);
    });
}

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
                         float translateZ) {
    const std::size_t count = inX.size();
    if (inY.size() != count || inZ.size() != count || outX.size() != count || outY.size() != count ||
        outZ.size() != count) {
        throw std::invalid_argument("batchScaleTranslate: all spans must have identical size");
    }

    scaleTranslateSpanAxis(inX, outX, scaleX, translateX);
    scaleTranslateSpanAxis(inY, outY, scaleY, translateY);
    scaleTranslateSpanAxis(inZ, outZ, scaleZ, translateZ);
}

void batchScaleTranslate(AlignedSoaColumn<float>& x,
                         AlignedSoaColumn<float>& y,
                         AlignedSoaColumn<float>& z,
                         float scaleX,
                         float scaleY,
                         float scaleZ,
                         float translateX,
                         float translateY,
                         float translateZ) {
    const std::size_t count = x.size();
    if (y.size() != count || z.size() != count) {
        throw std::invalid_argument("batchScaleTranslate: all columns must have identical size");
    }
    if (count == 0) {
        return;
    }

    scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, x.assumeAlignedData(), scaleX, translateX);
    scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, y.assumeAlignedData(), scaleY, translateY);
    scaleTranslateInPlaceAxis<kSoaCacheLineBytes>(count, z.assumeAlignedData(), scaleZ, translateZ);
}

} // namespace lumora
