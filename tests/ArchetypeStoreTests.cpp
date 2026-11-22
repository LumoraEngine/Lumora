#include "core/ArchetypeStore.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lumora;

namespace {

struct alignas(128) OverAlignedItem {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::array<std::uint64_t, 14> padding{};
};

struct TrackedObject {
    inline static int aliveCount = 0;

    int value = 0;
    std::string label;

    TrackedObject() : value(0) { ++aliveCount; }
    TrackedObject(int v, std::string text) : value(v), label(std::move(text)) { ++aliveCount; }
    TrackedObject(const TrackedObject& other) : value(other.value), label(other.label) { ++aliveCount; }
    TrackedObject(TrackedObject&& other) noexcept : value(other.value), label(std::move(other.label)) {
        ++aliveCount;
    }
    TrackedObject& operator=(const TrackedObject& other) = default;
    TrackedObject& operator=(TrackedObject&& other) noexcept = default;
    ~TrackedObject() { --aliveCount; }
};

KinematicData makeTestEntityData(std::size_t index) {
    const float f = static_cast<float>(index + 1);
    const float sign = (index % 2 == 0) ? 1.0f : -1.0f;
    return KinematicData{
        .posX = f * 1.25f * sign,
        .posY = -f * 0.5f + 3.0f,
        .posZ = f * 0.75f - 10.0f,
        .velX = sign * (0.25f * f - 4.0f),
        .velY = 1.5f - 0.125f * f,
        .velZ = -sign * (0.5f + 0.0625f * f),
        .accX = 0.3f * sign,
        .accY = -9.81f,
        .accZ = 0.15f * f * sign,
        .scaleX = (index % 3 == 0) ? -1.75f : 1.5f,
        .scaleY = (index % 5 == 0) ? -0.0f : (0.5f + 0.01f * f),
        .scaleZ = (index % 4 == 0) ? -2.25f : 0.75f,
        .halfExtX = 0.5f + 0.05f * f,
        .halfExtY = 1.0f + 0.02f * f,
        .halfExtZ = 0.25f + 0.03f * f,
    };
}

template <class T>
void expect64ByteAligned(const AlignedSoaColumn<T>& column) {
    const auto addr = reinterpret_cast<std::uintptr_t>(column.data());
    EXPECT_EQ(addr % 64u, 0u);
    EXPECT_TRUE(isCacheLineAligned(column.data()));
}

} // namespace

TEST(AlignedSoaColumn, EmptyStateAndBoundsChecks) {
    AlignedSoaColumn<float> col;
    EXPECT_TRUE(col.empty());
    EXPECT_EQ(col.size(), 0u);
    EXPECT_EQ(col.capacity(), 0u);
    EXPECT_EQ(col.data(), nullptr);
    expect64ByteAligned(col);

    col.clear();
    col.shrinkToFit();
    col.reserve(0);
    EXPECT_EQ(col.data(), nullptr);
    EXPECT_THROW(col.swapRemove(0), std::out_of_range);
    EXPECT_THROW((void)col.at(0), std::out_of_range);
}

TEST(AlignedSoaColumn, Guarantees64ByteAlignmentAcrossAllMutations) {
    AlignedSoaColumn<float> floats;
    for (int i = 0; i < 200; ++i) {
        floats.pushBack(static_cast<float>(i));
        expect64ByteAligned(floats);
    }

    floats.reserve(1024);
    expect64ByteAligned(floats);
    EXPECT_GE(floats.capacity(), 1024u);

    floats.resize(33);
    expect64ByteAligned(floats);
    EXPECT_EQ(floats.size(), 33u);

    floats.shrinkToFit();
    expect64ByteAligned(floats);
    EXPECT_EQ(floats.capacity(), 33u);

    AlignedSoaColumn<float> copied(floats);
    expect64ByteAligned(copied);
    EXPECT_EQ(copied.size(), 33u);

    AlignedSoaColumn<float> moved(std::move(copied));
    expect64ByteAligned(moved);
    EXPECT_EQ(moved.size(), 33u);
    EXPECT_EQ(copied.size(), 0u);
    EXPECT_EQ(copied.data(), nullptr);

    AlignedSoaColumn<std::uint64_t> ids;
    for (std::uint64_t i = 0; i < 70; ++i) {
        ids.pushBack(i + 100);
        expect64ByteAligned(ids);
    }

    AlignedSoaColumn<OverAlignedItem> overAligned;
    for (std::uint64_t i = 0; i < 19; ++i) {
        overAligned.pushBack(OverAlignedItem{i, i * 2, {}});
        expect64ByteAligned(overAligned);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(overAligned.data()) % 128u, 0u);
    }
}

TEST(AlignedSoaColumn, SwapRemoveClearShrinkToFitAndSelfPush) {
    AlignedSoaColumn<int> col;
    col.reserve(4);
    col.pushBack(10);
    col.pushBack(20);
    col.pushBack(30);
    col.pushBack(40);
    ASSERT_EQ(col.capacity(), 4u);

    // Trigger reallocation while pushing a reference into existing storage.
    col.pushBack(col[0]);
    ASSERT_EQ(col.size(), 5u);
    EXPECT_EQ(col[4], 10);

    // Swap-remove middle element (index 1 -> replaced by back element 10).
    col.swapRemove(1);
    ASSERT_EQ(col.size(), 4u);
    EXPECT_EQ(col[0], 10);
    EXPECT_EQ(col[1], 10);
    EXPECT_EQ(col[2], 30);
    EXPECT_EQ(col[3], 40);

    // Swap-remove last element.
    col.swapRemove(3);
    ASSERT_EQ(col.size(), 3u);
    EXPECT_EQ(col[2], 30);

    const std::size_t capBeforeClear = col.capacity();
    col.clear();
    EXPECT_TRUE(col.empty());
    EXPECT_EQ(col.capacity(), capBeforeClear);

    col.shrinkToFit();
    EXPECT_EQ(col.capacity(), 0u);
    EXPECT_EQ(col.data(), nullptr);
}

TEST(AlignedSoaColumn, NonTrivialLifetimeAccounting) {
    TrackedObject::aliveCount = 0;
    {
        AlignedSoaColumn<TrackedObject> col;
        for (int i = 0; i < 35; ++i) {
            col.emplaceBack(i, "item_" + std::to_string(i));
        }
        EXPECT_EQ(TrackedObject::aliveCount, 35);

        col.swapRemove(0);
        col.swapRemove(10);
        col.swapRemove(col.size() - 1);
        EXPECT_EQ(TrackedObject::aliveCount, 32);

        AlignedSoaColumn<TrackedObject> copy = col;
        EXPECT_EQ(TrackedObject::aliveCount, 64);

        copy.resize(10);
        EXPECT_EQ(TrackedObject::aliveCount, 42);

        copy.clear();
        EXPECT_EQ(TrackedObject::aliveCount, 32);
    }
    EXPECT_EQ(TrackedObject::aliveCount, 0);
}

TEST(KinematicSoaTable, All16ColumnsAre64ByteAligned) {
    KinematicSoaTable table(ComponentTag::AllKinematic);
    for (std::size_t i = 0; i < 129; ++i) {
        ASSERT_TRUE(table.addEntity(i, makeTestEntityData(i)));
    }

    expect64ByteAligned(table.entityIds());
    expect64ByteAligned(table.posX());
    expect64ByteAligned(table.posY());
    expect64ByteAligned(table.posZ());
    expect64ByteAligned(table.velX());
    expect64ByteAligned(table.velY());
    expect64ByteAligned(table.velZ());
    expect64ByteAligned(table.accX());
    expect64ByteAligned(table.accY());
    expect64ByteAligned(table.accZ());
    expect64ByteAligned(table.scaleX());
    expect64ByteAligned(table.scaleY());
    expect64ByteAligned(table.scaleZ());
    expect64ByteAligned(table.halfExtX());
    expect64ByteAligned(table.halfExtY());
    expect64ByteAligned(table.halfExtZ());
}

TEST(KinematicSoaTable, SwapAndPopLookupStabilityAndPartialColumns) {
    KinematicSoaTable partial(ComponentTag::Position | ComponentTag::Velocity);
    EXPECT_TRUE(partial.addEntity(0, makeTestEntityData(0)));
    EXPECT_FALSE(partial.addEntity(0, makeTestEntityData(1)));  // duplicate rejected

    EXPECT_EQ(partial.posX().size(), 1u);
    EXPECT_EQ(partial.velX().size(), 1u);
    EXPECT_EQ(partial.accX().size(), 0u);
    EXPECT_EQ(partial.accX().capacity(), 0u);
    EXPECT_EQ(partial.scaleX().capacity(), 0u);
    EXPECT_EQ(partial.halfExtX().capacity(), 0u);

    constexpr std::size_t kCount = 250;
    for (std::size_t i = 1; i < kCount; ++i) {
        ASSERT_TRUE(partial.addEntity(i * 10, makeTestEntityData(i)));
    }
    ASSERT_EQ(partial.size(), kCount);

    // Remove first, middle, even, and trailing entities in varied order.
    std::vector<EntityId> remaining;
    for (std::size_t i = 0; i < kCount; ++i) {
        const EntityId id = i == 0 ? 0 : i * 10;
        if (i == 0 || i == kCount - 1 || i % 3 == 0 || i == 77) {
            EXPECT_TRUE(partial.removeEntity(id));
            EXPECT_FALSE(partial.removeEntity(id));
        } else {
            remaining.push_back(id);
        }
    }

    EXPECT_EQ(partial.size(), remaining.size());
    EXPECT_EQ(partial.entityIndexMap().size(), remaining.size());

    for (EntityId id : remaining) {
        const auto idx = partial.indexOf(id);
        ASSERT_TRUE(idx.has_value());
        ASSERT_LT(*idx, partial.size());
        EXPECT_EQ(partial.entityIds()[*idx], id);
        EXPECT_EQ(partial.entityIndexMap().at(id), *idx);

        const std::size_t origIdx = static_cast<std::size_t>(id / 10);
        const KinematicData expected = makeTestEntityData(origIdx);
        const auto actual = partial.readEntity(id);
        ASSERT_TRUE(actual.has_value());
        EXPECT_FLOAT_EQ(actual->posX, expected.posX);
        EXPECT_FLOAT_EQ(actual->posY, expected.posY);
        EXPECT_FLOAT_EQ(actual->posZ, expected.posZ);
        EXPECT_FLOAT_EQ(actual->velX, expected.velX);
        EXPECT_FLOAT_EQ(actual->velY, expected.velY);
        EXPECT_FLOAT_EQ(actual->velZ, expected.velZ);
    }

    // Drain remaining elements from front to verify single-element pop stability.
    for (EntityId id : remaining) {
        ASSERT_TRUE(partial.removeEntity(id));
    }
    EXPECT_TRUE(partial.empty());
    EXPECT_TRUE(partial.entityIndexMap().empty());
}

TEST(ArchetypeRegistry, MigrationPreservesExistingFieldsAndLookupStability) {
    ArchetypeRegistry registry;

    const ArchetypeMask posVel = ComponentTag::Position | ComponentTag::Velocity | ComponentTag::Custom0;
    const ArchetypeMask full = ComponentTag::AllKinematic | ComponentTag::Custom1;
    const ArchetypeMask posScaleBounds = ComponentTag::Position | ComponentTag::Scale | ComponentTag::Bounds;

    for (EntityId id = 1; id <= 12; ++id) {
        KinematicData init = makeTestEntityData(static_cast<std::size_t>(id));
        ASSERT_TRUE(registry.addEntity(id, posVel, init));
    }
    EXPECT_EQ(registry.entityCount(), 12u);
    EXPECT_EQ(registry.archetypeCount(), 1u);

    // Migrate entity 1 (front), 6 (middle), and 12 (back) to `full`.
    for (EntityId target : {EntityId{1}, EntityId{6}, EntityId{12}}) {
        const KinematicData orig = makeTestEntityData(static_cast<std::size_t>(target));
        ASSERT_TRUE(registry.migrateEntity(target, full));
        EXPECT_EQ(registry.archetypeOf(target), full);

        const auto migrated = registry.readEntity(target);
        ASSERT_TRUE(migrated.has_value());
        EXPECT_FLOAT_EQ(migrated->posX, orig.posX);
        EXPECT_FLOAT_EQ(migrated->posY, orig.posY);
        EXPECT_FLOAT_EQ(migrated->posZ, orig.posZ);
        EXPECT_FLOAT_EQ(migrated->velX, orig.velX);
        EXPECT_FLOAT_EQ(migrated->velY, orig.velY);
        EXPECT_FLOAT_EQ(migrated->velZ, orig.velZ);
        // Newly added components default-initialized.
        EXPECT_FLOAT_EQ(migrated->accX, 0.0f);
        EXPECT_FLOAT_EQ(migrated->scaleX, 1.0f);
        EXPECT_FLOAT_EQ(migrated->scaleY, 1.0f);
        EXPECT_FLOAT_EQ(migrated->scaleZ, 1.0f);
        EXPECT_FLOAT_EQ(migrated->halfExtX, 0.0f);

        // Populate newly added components and migrate again dropping Velocity & Acceleration.
        ASSERT_TRUE(registry.writeEntity(target, orig));
        ASSERT_TRUE(registry.migrateEntity(target, posScaleBounds));

        const auto afterDrop = registry.readEntity(target);
        ASSERT_TRUE(afterDrop.has_value());
        EXPECT_FLOAT_EQ(afterDrop->posX, orig.posX);
        EXPECT_FLOAT_EQ(afterDrop->scaleX, orig.scaleX);
        EXPECT_FLOAT_EQ(afterDrop->scaleY, orig.scaleY);
        EXPECT_FLOAT_EQ(afterDrop->scaleZ, orig.scaleZ);
        EXPECT_FLOAT_EQ(afterDrop->halfExtX, orig.halfExtX);
        EXPECT_FLOAT_EQ(afterDrop->halfExtY, orig.halfExtY);
        EXPECT_FLOAT_EQ(afterDrop->halfExtZ, orig.halfExtZ);
        EXPECT_FLOAT_EQ(afterDrop->velX, 0.0f);
        EXPECT_FLOAT_EQ(afterDrop->accX, 0.0f);
    }

    // Verify all remaining entities in `posVel` kept intact O(1) indices after swap-and-pop migrations.
    const KinematicSoaTable* srcTable = registry.findTable(posVel);
    ASSERT_NE(srcTable, nullptr);
    EXPECT_EQ(srcTable->size(), 9u);
    for (EntityId id = 2; id <= 11; ++id) {
        if (id == 6) {
            EXPECT_FALSE(srcTable->contains(id));
            continue;
        }
        EXPECT_EQ(registry.findEntityTable(id), srcTable);
        const auto row = srcTable->indexOf(id);
        ASSERT_TRUE(row.has_value());
        EXPECT_EQ(srcTable->entityIds()[*row], id);

        const KinematicData expected = makeTestEntityData(static_cast<std::size_t>(id));
        const auto actual = registry.readEntity(id);
        ASSERT_TRUE(actual.has_value());
        EXPECT_FLOAT_EQ(actual->posX, expected.posX);
        EXPECT_FLOAT_EQ(actual->velZ, expected.velZ);
    }

    // `full` table became empty after migrating its 3 entities to `posScaleBounds`.
    const KinematicSoaTable* emptyFull = registry.findTable(full);
    ASSERT_NE(emptyFull, nullptr);
    EXPECT_TRUE(emptyFull->empty());
    registry.pruneEmptyTables();
    EXPECT_EQ(registry.findTable(full), nullptr);
    EXPECT_EQ(registry.archetypeCount(), 2u);
}

TEST(ArchetypeRegistry, AddComponentsAndForEachArchetypeFiltering) {
    ArchetypeRegistry registry;
    const ArchetypeMask customMask = ComponentTag::Position | ComponentTag::customBit(20);

    ASSERT_TRUE(registry.addEntity(100, ComponentTag::Position, KinematicData{.posX = 4.0f, .posY = 5.0f, .posZ = 6.0f}));
    ASSERT_TRUE(registry.addEntity(200, customMask, KinematicData{.posX = -2.0f}));
    ASSERT_TRUE(registry.addEntity(300, ComponentTag::Position | ComponentTag::Velocity));

    // Add Velocity & Acceleration onto entity 100 without clobbering existing Position.
    ASSERT_TRUE(registry.addComponents(
        100, ComponentTag::Position | ComponentTag::Velocity | ComponentTag::Acceleration,
        KinematicData{.posX = 999.0f, .velX = 10.0f, .accX = 2.0f}));

    const auto e100 = registry.readEntity(100);
    ASSERT_TRUE(e100.has_value());
    EXPECT_FLOAT_EQ(e100->posX, 4.0f);
    EXPECT_FLOAT_EQ(e100->velX, 10.0f);
    EXPECT_FLOAT_EQ(e100->accX, 2.0f);

    std::size_t visitedPosVel = 0;
    std::size_t entitySumPosVel = 0;
    forEachArchetype(registry, ComponentTag::Position | ComponentTag::Velocity,
                     [&](const KinematicSoaTable& table) {
                         ++visitedPosVel;
                         entitySumPosVel += table.size();
                     });
    EXPECT_EQ(visitedPosVel, 2u);
    EXPECT_EQ(entitySumPosVel, 2u);

    std::size_t visitedCustom = 0;
    registry.forEachArchetype(ComponentTag::customBit(20), [&](const KinematicSoaTable& table) {
        ++visitedCustom;
        EXPECT_EQ(table.size(), 1u);
        EXPECT_EQ(table.entityIds()[0], 200u);
    });
    EXPECT_EQ(visitedCustom, 1u);
}

TEST(BatchKernels, IntegrateKinematicsMatchesScalarAcrossNonMultipleOf8Sizes) {
    const std::array<std::size_t, 9> sizes = {0, 1, 3, 4, 7, 8, 15, 65, 1027};
    constexpr float kDt = 1.0f / 60.0f;

    for (std::size_t count : sizes) {
        KinematicSoaTable table(ComponentTag::Position | ComponentTag::Velocity | ComponentTag::Acceleration);
        std::vector<KinematicData> ref(count);

        for (std::size_t i = 0; i < count; ++i) {
            ref[i] = makeTestEntityData(i);
            ASSERT_TRUE(table.addEntity(i, ref[i]));
        }

        for (int step = 0; step < 3; ++step) {
            table.integrateKinematics(kDt);
            for (std::size_t i = 0; i < count; ++i) {
                ref[i].velX += ref[i].accX * kDt;
                ref[i].velY += ref[i].accY * kDt;
                ref[i].velZ += ref[i].accZ * kDt;
                ref[i].posX += ref[i].velX * kDt;
                ref[i].posY += ref[i].velY * kDt;
                ref[i].posZ += ref[i].velZ * kDt;
            }
        }

        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_FLOAT_EQ(table.velX()[i], ref[i].velX) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(table.velY()[i], ref[i].velY) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(table.velZ()[i], ref[i].velZ) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(table.posX()[i], ref[i].posX) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(table.posY()[i], ref[i].posY) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(table.posZ()[i], ref[i].posZ) << "count=" << count << " i=" << i;
        }
    }

    // Also test partial Velocity+Position (no Acceleration) via ArchetypeRegistry.
    ArchetypeRegistry registry;
    for (std::size_t i = 0; i < 67; ++i) {
        ASSERT_TRUE(registry.addEntity(i, ComponentTag::Position | ComponentTag::Velocity, makeTestEntityData(i)));
    }
    registry.integrateKinematics(0.5f);
    for (std::size_t i = 0; i < 67; ++i) {
        const KinematicData orig = makeTestEntityData(i);
        const auto got = registry.readEntity(i);
        ASSERT_TRUE(got.has_value());
        EXPECT_FLOAT_EQ(got->velX, orig.velX);
        EXPECT_FLOAT_EQ(got->posX, orig.posX + orig.velX * 0.5f);
        EXPECT_FLOAT_EQ(got->posY, orig.posY + orig.velY * 0.5f);
        EXPECT_FLOAT_EQ(got->posZ, orig.posZ + orig.velZ * 0.5f);
    }
}

TEST(BatchKernels, ComputeWorldAabbsMatchesScalarAcrossNonMultipleOf8Sizes) {
    const std::array<std::size_t, 9> sizes = {0, 1, 3, 4, 7, 8, 15, 65, 1027};

    for (std::size_t count : sizes) {
        KinematicSoaTable table(ComponentTag::Position | ComponentTag::Scale | ComponentTag::Bounds);
        std::vector<KinematicData> ref(count);
        for (std::size_t i = 0; i < count; ++i) {
            ref[i] = makeTestEntityData(i);
            ASSERT_TRUE(table.addEntity(i, ref[i]));
        }

        AlignedSoaColumn<float> minX, minY, minZ, maxX, maxY, maxZ;
        table.computeWorldAabbs(minX, minY, minZ, maxX, maxY, maxZ);

        ASSERT_EQ(minX.size(), count);
        ASSERT_EQ(maxZ.size(), count);
        expect64ByteAligned(minX);
        expect64ByteAligned(minY);
        expect64ByteAligned(minZ);
        expect64ByteAligned(maxX);
        expect64ByteAligned(maxY);
        expect64ByteAligned(maxZ);

        for (std::size_t i = 0; i < count; ++i) {
            const float extX = ref[i].halfExtX * std::abs(ref[i].scaleX);
            const float extY = ref[i].halfExtY * std::abs(ref[i].scaleY);
            const float extZ = ref[i].halfExtZ * std::abs(ref[i].scaleZ);
            EXPECT_FLOAT_EQ(minX[i], ref[i].posX - extX) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(minY[i], ref[i].posY - extY) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(minZ[i], ref[i].posZ - extZ) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(maxX[i], ref[i].posX + extX) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(maxY[i], ref[i].posY + extY) << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(maxZ[i], ref[i].posZ + extZ) << "count=" << count << " i=" << i;
        }
    }

    // Table without Scale uses implicit scale = 1.0f.
    KinematicSoaTable unscaled(ComponentTag::Position | ComponentTag::Bounds);
    for (std::size_t i = 0; i < 19; ++i) {
        ASSERT_TRUE(unscaled.addEntity(i, makeTestEntityData(i)));
    }
    AlignedSoaColumn<float> minX, minY, minZ, maxX, maxY, maxZ;
    unscaled.computeWorldAabbs(minX, minY, minZ, maxX, maxY, maxZ);
    for (std::size_t i = 0; i < 19; ++i) {
        const KinematicData ref = makeTestEntityData(i);
        EXPECT_FLOAT_EQ(minX[i], ref.posX - ref.halfExtX);
        EXPECT_FLOAT_EQ(maxX[i], ref.posX + ref.halfExtX);
    }

    EXPECT_THROW(unscaled.computeWorldAabbs(minX, minX, minZ, maxX, maxY, maxZ), std::invalid_argument);
}

TEST(BatchKernels, BatchScaleTranslateMatchesScalarAcrossVariantsAndSizes) {
    const std::array<std::size_t, 9> sizes = {0, 1, 3, 4, 7, 8, 15, 65, 1027};
    constexpr float sx = 2.5f, sy = -0.75f, sz = 1.25f;
    constexpr float tx = 10.0f, ty = -3.5f, tz = 0.25f;

    for (std::size_t count : sizes) {
        KinematicSoaTable table(ComponentTag::Position);
        AlignedSoaColumn<float> inX, inY, inZ, outX, outY, outZ;
        outX.resize(count);
        outY.resize(count);
        outZ.resize(count);

        std::vector<KinematicData> ref(count);
        for (std::size_t i = 0; i < count; ++i) {
            ref[i] = makeTestEntityData(i);
            ASSERT_TRUE(table.addEntity(i, ref[i]));
            inX.pushBack(ref[i].posX);
            inY.pushBack(ref[i].posY);
            inZ.pushBack(ref[i].posZ);
        }

        table.batchScaleTranslate(sx, sy, sz, tx, ty, tz);
        batchScaleTranslate(inX.span(), inY.span(), inZ.span(), outX.span(), outY.span(), outZ.span(),
                            sx, sy, sz, tx, ty, tz);
        batchScaleTranslate(inX, inY, inZ, sx, sy, sz, tx, ty, tz);

        for (std::size_t i = 0; i < count; ++i) {
            const float expX = ref[i].posX * sx + tx;
            const float expY = ref[i].posY * sy + ty;
            const float expZ = ref[i].posZ * sz + tz;
            EXPECT_FLOAT_EQ(table.posX()[i], expX);
            EXPECT_FLOAT_EQ(table.posY()[i], expY);
            EXPECT_FLOAT_EQ(table.posZ()[i], expZ);
            EXPECT_FLOAT_EQ(outX[i], expX);
            EXPECT_FLOAT_EQ(outY[i], expY);
            EXPECT_FLOAT_EQ(outZ[i], expZ);
            EXPECT_FLOAT_EQ(inX[i], expX);
            EXPECT_FLOAT_EQ(inY[i], expY);
            EXPECT_FLOAT_EQ(inZ[i], expZ);
        }
    }

    // Unaligned std::span offset + overlapping span fallback check.
    std::vector<float> bufX(70), bufY(70), bufZ(70);
    std::iota(bufX.begin(), bufX.end(), 1.0f);
    std::iota(bufY.begin(), bufY.end(), 101.0f);
    std::iota(bufZ.begin(), bufZ.end(), 201.0f);

    const std::vector<float> origX = bufX;
    // Overlapping shift right by 1 element (requires backward traversal to avoid overwriting unread inputs).
    batchScaleTranslate(std::span<const float>(bufX.data(), 65),
                        std::span<const float>(bufY.data(), 65),
                        std::span<const float>(bufZ.data(), 65),
                        std::span<float>(bufX.data() + 1, 65),
                        std::span<float>(bufY.data() + 1, 65),
                        std::span<float>(bufZ.data() + 1, 65),
                        sx, sy, sz, tx, ty, tz);

    for (std::size_t i = 0; i < 65; ++i) {
        EXPECT_FLOAT_EQ(bufX[i + 1], origX[i] * sx + tx);
    }
}
