#include "scene/WorldStreaming.hpp"

#include "TestUtils.hpp"
#include "scene/Components.hpp"
#include "scene/Scene.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace lumora;
using namespace std::chrono_literals;

namespace {

SectorEntityRecord makeSampleRecord(std::uint64_t uuidVal, const std::string& name, glm::vec3 pos,
                                    HlodLevel minLod = HlodLevel::Lod2Coarse,
                                    std::string mesh = "Content/Models/Rock.glb#0",
                                    std::string mat = "Content/Materials/Stone.lmat") {
    SectorEntityRecord rec;
    rec.id = Uuid(uuidVal);
    rec.name = name;
    rec.position = pos;
    rec.rotationDeg = {0.0f, 45.0f, 0.0f};
    rec.scale = {1.0f, 1.5f, 1.0f};
    rec.meshAsset = std::move(mesh);
    rec.materialAsset = std::move(mat);
    rec.boundsMin = pos - glm::vec3(1.0f);
    rec.boundsMax = pos + glm::vec3(1.0f);
    rec.minLod = minLod;
    return rec;
}

SectorPackage makeIncompressiblePackage(std::uint32_t seed) {
    std::mt19937 rng(seed);
    auto makeNoise = [&](std::size_t len) {
        std::string s(len, '\0');
        for (char& ch : s) {
            ch = static_cast<char>('!' + (rng() % 90u));
        }
        return s;
    };
    SectorPackage pkg;
    pkg.coord = {2, -3};
    pkg.boundsMin = {1.13f, 2.27f, 3.41f};
    pkg.boundsMax = {4.53f, 5.67f, 6.79f};
    SectorEntityRecord rec;
    rec.id = Uuid(0x9876543210FEDCBAull ^ static_cast<std::uint64_t>(seed));
    rec.name = makeNoise(96);
    rec.position = {11.17f, 22.31f, 33.47f};
    rec.rotationDeg = {12.34f, 56.78f, 90.12f};
    rec.scale = {1.11f, 2.22f, 3.33f};
    rec.meshAsset = makeNoise(96);
    rec.materialAsset = makeNoise(96);
    rec.boundsMin = {7.19f, 8.23f, 9.29f};
    rec.boundsMax = {14.37f, 15.43f, 16.49f};
    rec.minLod = HlodLevel::Lod0Full;
    pkg.entities.push_back(std::move(rec));
    return pkg;
}

template <class Pred>
bool waitUntil(Pred&& pred, std::chrono::milliseconds timeout = 4000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(2ms);
    }
    return true;
}

bool waitAndCommit(WorldStreamer& streamer, Scene& scene, std::size_t expectedActiveSectors,
                   std::chrono::milliseconds timeout = 4000ms) {
    return waitUntil(
        [&]() {
            streamer.commitReadySectors(scene);
            const StreamingStats st = streamer.stats();
            return st.loadingSectors == 0u && st.stagedSectors == 0u && st.activeSectors == expectedActiveSectors;
        },
        timeout);
}

} // namespace

TEST(WorldStreaming, SectorCoordHashAndGridMapping) {
    std::unordered_set<SectorCoord> coords;
    for (std::int32_t z = -16; z < 16; ++z) {
        for (std::int32_t x = -16; x < 16; ++x) {
            EXPECT_TRUE(coords.insert(SectorCoord{x, z}).second);
        }
    }
    EXPECT_EQ(coords.size(), 1024u);
    EXPECT_EQ((SectorCoord{3, -4}), (SectorCoord{3, -4}));
    EXPECT_NE((SectorCoord{3, -4}), (SectorCoord{-4, 3}));

    EXPECT_EQ(worldToSectorCoord({0.0f, 0.0f, 0.0f}, 128.0f), (SectorCoord{0, 0}));
    EXPECT_EQ(worldToSectorCoord({127.9f, 50.0f, 127.9f}, 128.0f), (SectorCoord{0, 0}));
    EXPECT_EQ(worldToSectorCoord({128.0f, 0.0f, 0.0f}, 128.0f), (SectorCoord{1, 0}));
    EXPECT_EQ(worldToSectorCoord({-0.01f, 0.0f, -128.1f}, 128.0f), (SectorCoord{-1, -2}));

    EXPECT_FLOAT_EQ(distanceToSectorXZ({64.0f, 10.0f, 64.0f}, {0, 0}, 128.0f), 0.0f);
    EXPECT_FLOAT_EQ(distanceToSectorXZ({138.0f, 0.0f, 64.0f}, {0, 0}, 128.0f), 10.0f);
    EXPECT_NEAR(distanceToSectorXZ({131.0f, 0.0f, 132.0f}, {0, 0}, 128.0f), 5.0f, 1e-4f);
}

TEST(WorldStreaming, EvaluateSectorHlodRespectsHysteresis) {
    StreamingConfig cfg;
    cfg.lod0Radius = 150.0f;
    cfg.lod1Radius = 350.0f;
    cfg.lod2Radius = 700.0f;
    cfg.hysteresis = 16.0f;

    // Entering from Unloaded uses exact radii.
    EXPECT_EQ(evaluateSectorHlod(150.0f, HlodLevel::Unloaded, cfg), HlodLevel::Lod0Full);
    EXPECT_EQ(evaluateSectorHlod(200.0f, HlodLevel::Unloaded, cfg), HlodLevel::Lod1Medium);
    EXPECT_EQ(evaluateSectorHlod(700.0f, HlodLevel::Unloaded, cfg), HlodLevel::Lod2Coarse);
    EXPECT_EQ(evaluateSectorHlod(705.0f, HlodLevel::Unloaded, cfg), HlodLevel::Unloaded);

    // Exiting Lod0/Lod1/Lod2 uses radius + hysteresis.
    EXPECT_EQ(evaluateSectorHlod(160.0f, HlodLevel::Lod0Full, cfg), HlodLevel::Lod0Full);
    EXPECT_EQ(evaluateSectorHlod(167.0f, HlodLevel::Lod0Full, cfg), HlodLevel::Lod1Medium);
    EXPECT_EQ(evaluateSectorHlod(160.0f, HlodLevel::Lod1Medium, cfg), HlodLevel::Lod1Medium);

    EXPECT_EQ(evaluateSectorHlod(715.0f, HlodLevel::Lod2Coarse, cfg), HlodLevel::Lod2Coarse);
    EXPECT_EQ(evaluateSectorHlod(717.0f, HlodLevel::Lod2Coarse, cfg), HlodLevel::Unloaded);
}

TEST(WorldStreaming, LzBlockCompressionRoundtripAndLongMatches) {
    // Empty block
    const auto emptyComp = compressSectorBlock({});
    EXPECT_TRUE(emptyComp.empty());
    const auto emptyDecomp = decompressSectorBlock(emptyComp, 0u);
    ASSERT_TRUE(emptyDecomp.has_value());
    EXPECT_TRUE(emptyDecomp->empty());

    // Short (< 12 bytes) block
    std::vector<std::byte> tiny{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
    const auto tinyComp = compressSectorBlock(tiny);
    const auto tinyDecomp = decompressSectorBlock(tinyComp, tiny.size());
    ASSERT_TRUE(tinyDecomp.has_value()) << tinyDecomp.error().message;
    EXPECT_EQ(*tinyDecomp, tiny);

    // Large repetitive block exercising >255 literal and >255 match varints.
    std::vector<std::byte> large;
    for (int i = 0; i < 400; ++i) {
        large.push_back(static_cast<std::byte>((i * 37) ^ (i >> 2)));
    }
    const std::size_t patternStart = large.size();
    for (int i = 0; i < 1200; ++i) {
        large.push_back(std::byte{0x5A});
    }
    for (std::size_t i = 0; i < patternStart; ++i) {
        large.push_back(large[i]);
    }
    const auto largeComp = compressSectorBlock(large);
    EXPECT_LT(largeComp.size(), large.size());
    const auto largeDecomp = decompressSectorBlock(largeComp, large.size());
    ASSERT_TRUE(largeDecomp.has_value()) << largeDecomp.error().message;
    EXPECT_EQ(*largeDecomp, large);

    // Malformed LZ stream checks
    EXPECT_FALSE(decompressSectorBlock({}, 10u).has_value());
    EXPECT_FALSE(decompressSectorBlock(tinyComp, 0u).has_value());
    std::vector<std::byte> badOffset{std::byte{0x10}, std::byte{0xAB}, std::byte{0x05}, std::byte{0x00}};
    EXPECT_FALSE(decompressSectorBlock(badOffset, 16u).has_value());
}

TEST(WorldStreaming, SectorPackageCompressRoundtripAndRawFallback) {
    test::TempDir dir;

    // 1. Compressible sector with repeated assets/names
    SectorPackage pkg;
    pkg.coord = SectorCoord{-3, 7};
    pkg.boundsMin = {-384.0f, -10.0f, 896.0f};
    pkg.boundsMax = {-256.0f, 120.0f, 1024.0f};
    for (int i = 0; i < 64; ++i) {
        pkg.entities.push_back(makeSampleRecord(
            1000u + static_cast<std::uint64_t>(i),
            "CliffRock",
            {static_cast<float>(i) * 2.0f, 1.5f, 900.0f + static_cast<float>(i % 4)},
            static_cast<HlodLevel>(i % 3)));
    }

    const auto encoded = encodeSectorPackage(pkg);
    ASSERT_TRUE(encoded.has_value()) << encoded.error().message;
    // Compressed flag (bit 0) must be set at header offset 8..11.
    EXPECT_EQ(static_cast<std::uint8_t>((*encoded)[8]) & 1u, 1u);

    const std::filesystem::path file = sectorFilePath(dir.path(), pkg.coord);
    ASSERT_TRUE(saveSectorPackage(file, pkg).has_value());
    const auto loaded = loadSectorPackage(file);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    EXPECT_EQ(*loaded, pkg);

    // 2. Empty sector roundtrip
    SectorPackage emptyPkg;
    emptyPkg.coord = {0, 0};
    const auto emptyFile = dir.path() / "sector_0_0.lsector";
    ASSERT_TRUE(saveSectorPackage(emptyFile, emptyPkg).has_value());
    const auto loadedEmpty = loadSectorPackage(emptyFile);
    ASSERT_TRUE(loadedEmpty.has_value()) << loadedEmpty.error().message;
    EXPECT_EQ(*loadedEmpty, emptyPkg);

    // 3. Incompressible high-entropy payload triggers raw fallback flag (flags == 0)
    const SectorPackage randomPkg = makeIncompressiblePackage(0xDEADBEEFu);
    const auto rawEncoded = encodeSectorPackage(randomPkg);
    ASSERT_TRUE(rawEncoded.has_value()) << rawEncoded.error().message;
    EXPECT_EQ(static_cast<std::uint8_t>((*rawEncoded)[8]), 0u) << "incompressible payload should use raw fallback";
    const auto rawDecoded = decodeSectorPackage(*rawEncoded);
    ASSERT_TRUE(rawDecoded.has_value()) << rawDecoded.error().message;
    EXPECT_EQ(*rawDecoded, randomPkg);
}

TEST(WorldStreaming, SectorPackageDetectsAllCorruptionModes) {
    SectorPackage pkg;
    pkg.coord = {1, -2};
    for (int i = 0; i < 8; ++i) {
        pkg.entities.push_back(makeSampleRecord(500u + static_cast<std::uint64_t>(i),
                                                "Prop_" + std::to_string(i),
                                                {130.0f + static_cast<float>(i), 0.0f, -250.0f}));
    }

    const auto encodedRes = encodeSectorPackage(pkg);
    ASSERT_TRUE(encodedRes.has_value());
    const std::vector<std::byte>& valid = *encodedRes;
    ASSERT_GT(valid.size(), 40u);

    // Bad magic
    {
        auto corrupted = valid;
        corrupted[0] = std::byte{'X'};
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }
    // Unsupported version
    {
        auto corrupted = valid;
        corrupted[4] = std::byte{2};
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }
    // Unknown flags
    {
        auto corrupted = valid;
        corrupted[8] = std::byte{0x80};
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }
    // Truncated header & truncated payload
    {
        EXPECT_FALSE(decodeSectorPackage(std::span<const std::byte>(valid).subspan(0u, 16u)).has_value());
        EXPECT_FALSE(decodeSectorPackage(std::span<const std::byte>(valid).subspan(0u, valid.size() - 1u)).has_value());
    }
    // Trailing garbage byte
    {
        auto corrupted = valid;
        corrupted.push_back(std::byte{0});
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }
    // Flipped checksum byte (header bytes 20..27)
    {
        auto corrupted = valid;
        corrupted[22] ^= std::byte{0x01};
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }
    // Flipped payload byte (middle of compressed stream)
    {
        auto corrupted = valid;
        corrupted[valid.size() - 5u] ^= std::byte{0xFF};
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }

    // Flipped payload byte in raw uncompressed fallback mode
    const SectorPackage rawPkg = makeIncompressiblePackage(0x12345678u);
    const auto rawEncoded = encodeSectorPackage(rawPkg);
    ASSERT_TRUE(rawEncoded.has_value());
    ASSERT_EQ(static_cast<std::uint8_t>((*rawEncoded)[8]), 0u);
    {
        auto corrupted = *rawEncoded;
        corrupted[corrupted.size() - 10u] ^= std::byte{0x20};
        EXPECT_FALSE(decodeSectorPackage(corrupted).has_value());
    }

    // Corrupt file on disk via loadSectorPackage
    test::TempDir dir;
    const auto badPath = dir.path() / "sector_1_-2.lsector";
    {
        std::ofstream out(badPath, std::ios::binary);
        auto corrupted = valid;
        corrupted[24] ^= std::byte{0x55};
        out.write(reinterpret_cast<const char*>(corrupted.data()), static_cast<std::streamsize>(corrupted.size()));
    }
    EXPECT_FALSE(loadSectorPackage(badPath).has_value());
    EXPECT_FALSE(loadSectorPackage(dir.path() / "does_not_exist.lsector").has_value());
}

TEST(WorldStreaming, AsyncDiskStreamingUnloadAndHysteresisAntiThrashing) {
    test::TempDir dir;

    const std::vector<SectorCoord> gridCoords{
        {0, 0},
        {1, 0},
        {-1, 0},
        {0, 1},
    };

    std::uint64_t nextId = 100u;
    std::size_t totalRecords = 0;
    for (SectorCoord coord : gridCoords) {
        SectorPackage pkg;
        pkg.coord = coord;
        for (int i = 0; i < 3; ++i) {
            const glm::vec3 pos{
                static_cast<float>(coord.x) * 128.0f + 16.0f + static_cast<float>(i) * 10.0f,
                static_cast<float>(i) * 2.0f,
                static_cast<float>(coord.z) * 128.0f + 32.0f,
            };
            pkg.entities.push_back(makeSampleRecord(
                nextId++,
                std::format("Entity_{}_{}_{}", coord.x, coord.z, i),
                pos,
                HlodLevel::Lod2Coarse,
                "Content/Models/Tree.glb#0",
                "Content/Materials/Bark.lmat"));
            ++totalRecords;
        }
        ASSERT_TRUE(saveSectorPackage(sectorFilePath(dir.path(), coord), pkg).has_value());
    }

    StreamingConfig cfg;
    cfg.cellSize = 128.0f;
    cfg.lod0Radius = 150.0f;
    cfg.lod1Radius = 350.0f;
    cfg.lod2Radius = 700.0f;
    cfg.hysteresis = 16.0f;
    cfg.commitBudgetMs = 10.0f;

    Scene scene("OpenWorld");
    WorldStreamer streamer(dir.path(), cfg);

    // 1. Focus inside (0,0) -> all 4 sectors within lod0Radius stream in asynchronously.
    streamer.updateFocus({64.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 4u)) << "timed out waiting for initial 4 sectors";

    StreamingStats st = streamer.stats();
    EXPECT_EQ(st.activeSectors, 4u);
    EXPECT_EQ(st.activeEntities, totalRecords);
    EXPECT_EQ(st.loadsCompleted, 4u);
    EXPECT_EQ(st.unloadsCompleted, 0u);
    EXPECT_EQ(scene.entityCount(), totalRecords);

    // Verify entity tagging, UUID tracking, MeshRendererComponent, and WorldTransformComponent.
    const entt::entity firstEntity = scene.findByUuid(Uuid(100u));
    ASSERT_NE(firstEntity, entt::entity{entt::null});
    ASSERT_TRUE(scene.registry().all_of<StreamedSectorTag>(firstEntity));
    EXPECT_EQ(scene.registry().get<StreamedSectorTag>(firstEntity).coord, (SectorCoord{0, 0}));
    ASSERT_TRUE(scene.registry().all_of<MeshRendererComponent>(firstEntity));
    EXPECT_EQ(scene.registry().get<MeshRendererComponent>(firstEntity).meshAsset, "Content/Models/Tree.glb#0");
    EXPECT_EQ(scene.registry().get<MeshRendererComponent>(firstEntity).material, "Content/Materials/Bark.lmat");
    const glm::vec3 worldPos = glm::vec3(scene.registry().get<WorldTransformComponent>(firstEntity).matrix[3]);
    EXPECT_NEAR(worldPos.x, 16.0f, 1e-4f);
    EXPECT_NEAR(worldPos.y, 0.0f, 1e-4f);
    EXPECT_NEAR(worldPos.z, 32.0f, 1e-4f);

    // 2. Move focus far away -> all 4 sectors unload cleanly.
    streamer.updateFocus({5000.0f, 0.0f, 5000.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 0u)) << "timed out unloading sectors";

    st = streamer.stats();
    EXPECT_EQ(st.activeSectors, 0u);
    EXPECT_EQ(st.activeEntities, 0u);
    EXPECT_EQ(st.loadsCompleted, 4u);
    EXPECT_EQ(st.unloadsCompleted, 4u);
    EXPECT_EQ(scene.entityCount(), 0u);

    // 3. Move back toward (-1, 0) (X range [-128, 0]), stopping inside the hysteresis band (700, 716]
    //    at X = -128 - 708 = -836 (distance to (-1,0) is 708 > lod2Radius 700, <= 716).
    //    Because (-1,0) is currently Unloaded, it must NOT reload inside (700, 716]!
    streamer.updateFocus({-836.0f, 0.0f, 64.0f});
    std::this_thread::sleep_for(25ms);
    streamer.commitReadySectors(scene);
    st = streamer.stats();
    EXPECT_EQ(st.activeSectors, 0u);
    EXPECT_EQ(st.loadsCompleted, 4u);

    // 4. Cross strictly inside lod2Radius (X = -128 - 692 = -820 -> distance = 692 <= 700) to load (-1, 0).
    streamer.updateFocus({-820.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 1u));
    st = streamer.stats();
    EXPECT_EQ(st.activeSectors, 1u);
    EXPECT_EQ(st.activeEntities, 3u);
    EXPECT_EQ(st.loadsCompleted, 5u);
    EXPECT_EQ(st.unloadsCompleted, 4u);
    EXPECT_EQ(streamer.sectorHlod({-1, 0}), HlodLevel::Lod2Coarse);

    // 5. Oscillate repeatedly back and forth across lod2Radius = 700 within the hysteresis band [692..714]
    //    (distances 708, 695, 714, 698, 710, 692). Must produce ZERO unload/reload thrashing!
    const float oscillatingFocusX[] = {-836.0f, -823.0f, -842.0f, -826.0f, -838.0f, -820.0f};
    for (float fx : oscillatingFocusX) {
        streamer.updateFocus({fx, 0.0f, 64.0f});
        std::this_thread::sleep_for(5ms);
        streamer.commitReadySectors(scene);
        const StreamingStats osc = streamer.stats();
        EXPECT_EQ(osc.activeSectors, 1u);
        EXPECT_EQ(osc.activeEntities, 3u);
        EXPECT_EQ(osc.loadsCompleted, 5u);
        EXPECT_EQ(osc.unloadsCompleted, 4u);
    }

    // 6. Crossing beyond lod2Radius + hysteresis (distance = 720 -> X = -848) triggers clean unload.
    streamer.updateFocus({-848.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 0u));
    st = streamer.stats();
    EXPECT_EQ(st.activeSectors, 0u);
    EXPECT_EQ(st.unloadsCompleted, 5u);
    EXPECT_EQ(scene.entityCount(), 0u);
}

TEST(WorldStreaming, HlodProgressionFiltersEntitiesWithoutReloadingFromDisk) {
    test::TempDir dir;

    SectorPackage pkg;
    pkg.coord = {0, 0};
    pkg.entities.push_back(makeSampleRecord(2001u, "DetailCup", {64.0f, 0.0f, 64.0f}, HlodLevel::Lod0Full));
    pkg.entities.push_back(makeSampleRecord(2002u, "MediumFence", {64.0f, 0.0f, 64.0f}, HlodLevel::Lod1Medium));
    pkg.entities.push_back(makeSampleRecord(2003u, "CoarseTower", {64.0f, 0.0f, 64.0f}, HlodLevel::Lod2Coarse));
    ASSERT_TRUE(saveSectorPackage(sectorFilePath(dir.path(), {0, 0}), pkg).has_value());

    Scene scene("HlodTest");
    WorldStreamer streamer(dir.path());

    // Start at dist = 500 -> Lod2Coarse (only CoarseTower).
    streamer.updateFocus({628.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 1u));
    EXPECT_EQ(streamer.sectorHlod({0, 0}), HlodLevel::Lod2Coarse);
    EXPECT_EQ(streamer.stats().activeEntities, 1u);
    EXPECT_NE(scene.findByUuid(Uuid(2003u)), entt::entity{entt::null});
    EXPECT_EQ(scene.findByUuid(Uuid(2002u)), entt::entity{entt::null});
    EXPECT_EQ(scene.findByUuid(Uuid(2001u)), entt::entity{entt::null});

    // Move to dist = 250 -> Lod1Medium (CoarseTower + MediumFence, zero extra disk loads).
    streamer.updateFocus({378.0f, 0.0f, 64.0f});
    streamer.commitReadySectors(scene);
    EXPECT_EQ(streamer.sectorHlod({0, 0}), HlodLevel::Lod1Medium);
    EXPECT_EQ(streamer.stats().activeEntities, 2u);
    EXPECT_EQ(streamer.stats().loadsCompleted, 1u);
    EXPECT_NE(scene.findByUuid(Uuid(2002u)), entt::entity{entt::null});

    // Move inside (0,0) -> Lod0Full (all 3 entities).
    streamer.updateFocus({64.0f, 0.0f, 64.0f});
    streamer.commitReadySectors(scene);
    EXPECT_EQ(streamer.sectorHlod({0, 0}), HlodLevel::Lod0Full);
    EXPECT_EQ(streamer.stats().activeEntities, 3u);
    EXPECT_EQ(streamer.stats().loadsCompleted, 1u);
    EXPECT_NE(scene.findByUuid(Uuid(2001u)), entt::entity{entt::null});

    // Step back to dist = 500 -> downgrades cleanly to Lod2Coarse.
    streamer.updateFocus({628.0f, 0.0f, 64.0f});
    streamer.commitReadySectors(scene);
    EXPECT_EQ(streamer.sectorHlod({0, 0}), HlodLevel::Lod2Coarse);
    EXPECT_EQ(streamer.stats().activeEntities, 1u);
    EXPECT_EQ(scene.entityCount(), 1u);
}

TEST(WorldStreaming, CommitBudgetTimeSlicesLargeSectorAndShutdownIsClean) {
    test::TempDir dir;

    SectorPackage bigPkg;
    bigPkg.coord = {0, 0};
    constexpr std::size_t kEntityCount = 240u;
    for (std::size_t i = 0; i < kEntityCount; ++i) {
        bigPkg.entities.push_back(makeSampleRecord(
            5000u + i,
            "Inst_" + std::to_string(i),
            {static_cast<float>(i % 16) * 4.0f, 0.0f, static_cast<float>(i / 16) * 4.0f},
            HlodLevel::Lod2Coarse));
    }
    ASSERT_TRUE(saveSectorPackage(sectorFilePath(dir.path(), {0, 0}), bigPkg).has_value());

    // Write a corrupt sibling file to verify resilience.
    {
        std::ofstream corrupt(sectorFilePath(dir.path(), {1, 0}), std::ios::binary);
        corrupt << "NOT_A_VALID_SECTOR";
    }

    StreamingConfig cfg;
    cfg.commitBudgetMs = 0.0001f; // Tiny positive budget forces multi-frame incremental commit.

    Scene scene("BudgetScene");
    {
        WorldStreamer streamer(dir.path(), cfg);
        streamer.updateFocus({64.0f, 0.0f, 64.0f});

        // Wait for background worker to finish I/O and decompression before committing.
        ASSERT_TRUE(waitUntil([&]() {
            const StreamingStats st = streamer.stats();
            return st.loadingSectors == 0u && st.stagedSectors == 1u;
        }));

        // First commit call should make forward progress without blocking indefinitely.
        streamer.commitReadySectors(scene);
        const StreamingStats mid = streamer.stats();
        EXPECT_GT(mid.activeEntities, 0u);
        EXPECT_LE(mid.activeEntities, kEntityCount);

        // Drain remaining entities across subsequent frames.
        int frames = 1;
        while (streamer.stats().stagedSectors > 0u && frames < 1000) {
            streamer.commitReadySectors(scene);
            ++frames;
        }
        const StreamingStats done = streamer.stats();
        EXPECT_EQ(done.stagedSectors, 0u);
        EXPECT_EQ(done.activeSectors, 1u);
        EXPECT_EQ(done.activeEntities, kEntityCount);
        EXPECT_EQ(scene.entityCount(), kEntityCount);

        // Enqueue work right before RAII destruction to exercise clean shutdown.
        streamer.updateFocus({2000.0f, 0.0f, 2000.0f});
    }
}

TEST(WorldStreaming, TimeSlicedLodFlipAndInvalidateSectorWorkCleanly) {
    test::TempDir dir;

    SectorPackage pkg;
    pkg.coord = {0, 0};
    for (int i = 0; i < 60; ++i) {
        const HlodLevel lod = (i % 2 == 0) ? HlodLevel::Lod0Full : HlodLevel::Lod2Coarse;
        pkg.entities.push_back(makeSampleRecord(7000u + static_cast<std::uint64_t>(i),
                                                "Flip_" + std::to_string(i),
                                                {32.0f, 0.0f, 32.0f}, lod));
    }
    ASSERT_TRUE(saveSectorPackage(sectorFilePath(dir.path(), {0, 0}), pkg).has_value());

    StreamingConfig cfg;
    cfg.commitBudgetMs = 0.0f; // Unlimited first to load all 60 at Lod0Full.
    Scene scene("LodFlipScene");
    WorldStreamer streamer(dir.path(), cfg);

    streamer.updateFocus({64.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 1u));
    EXPECT_EQ(streamer.stats().activeEntities, 60u);

    // Switch to tiny budget, downgrade to Lod1Medium (starts destroying Lod0 entities),
    // commit 1 slice mid-way, then immediately flip back to Lod0Full.
    cfg.commitBudgetMs = 0.0001f;
    streamer.setConfig(cfg);
    streamer.updateFocus({378.0f, 0.0f, 64.0f}); // dist = 250 -> Lod1Medium
    streamer.commitReadySectors(scene);
    EXPECT_LT(streamer.stats().activeEntities, 60u);

    streamer.updateFocus({64.0f, 0.0f, 64.0f}); // back to Lod0Full mid-reconcile
    for (int i = 0; i < 200; ++i) {
        streamer.commitReadySectors(scene);
    }
    EXPECT_EQ(streamer.stats().activeEntities, 60u);
    EXPECT_EQ(scene.entityCount(), 60u);

    // Absent sector created later is discovered once invalidateSector() clears the absent cache.
    streamer.updateFocus({64.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitUntil([&]() { return streamer.stats().loadingSectors == 0u; }));
    EXPECT_EQ(streamer.stats().activeSectors, 1u);

    SectorPackage latePkg;
    latePkg.coord = {1, 0};
    latePkg.entities.push_back(makeSampleRecord(8888u, "LateEntity", {150.0f, 0.0f, 32.0f}, HlodLevel::Lod2Coarse));
    ASSERT_TRUE(saveSectorPackage(sectorFilePath(dir.path(), {1, 0}), latePkg).has_value());

    streamer.invalidateSector({1, 0});
    streamer.updateFocus({64.0f, 0.0f, 64.0f});
    ASSERT_TRUE(waitAndCommit(streamer, scene, 2u));
    EXPECT_EQ(streamer.stats().activeEntities, 61u);
    EXPECT_NE(scene.findByUuid(Uuid(8888u)), entt::entity{entt::null});
}

