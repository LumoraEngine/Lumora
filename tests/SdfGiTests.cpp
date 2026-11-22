#include "renderer/SdfGi.hpp"

#include "platform/Paths.hpp"
#include "renderer/FrameResources.hpp"
#include "renderer/RenderContext.hpp"
#include "renderer/RenderGraph.hpp"
#include "renderer/TransientImagePool.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"
#include "scene/Camera.hpp"
#include "scene/Components.hpp"
#include "scene/Scene.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/matrix.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

using namespace lumora;
using namespace lumora::render;

namespace {

SdfGridConfig makeTestGridConfig(glm::uvec3 resolution = {48u, 48u, 48u}) {
    SdfGridConfig cfg;
    cfg.boundsMin = glm::vec3(-6.0f, -2.0f, -6.0f);
    cfg.boundsMax = glm::vec3(6.0f, 6.0f, 6.0f);
    cfg.resolution = resolution;
    cfg.maxSteps = 96u;
    cfg.maxDistance = 24.0f;
    cfg.bounceStrength = 1.0f;
    cfg.skyIntensity = 1.0f;
    cfg.raysPerPoint = 64u;
    return cfg;
}

} // namespace

TEST(SdfGi, ExactSignedDistanceBox) {
    const glm::vec3 boxMin(-1.0f, -1.0f, -1.0f);
    const glm::vec3 boxMax(1.0f, 1.0f, 1.0f);

    // Inside box (negative).
    EXPECT_FLOAT_EQ(signedDistanceBox({0.0f, 0.0f, 0.0f}, boxMin, boxMax), -1.0f);
    EXPECT_FLOAT_EQ(signedDistanceBox({0.5f, -0.25f, 0.1f}, boxMin, boxMax), -0.5f);
    EXPECT_LT(signedDistanceBox({0.99f, 0.0f, 0.0f}, boxMin, boxMax), 0.0f);

    // On box surface (exact 0).
    EXPECT_FLOAT_EQ(signedDistanceBox({1.0f, 0.0f, 0.0f}, boxMin, boxMax), 0.0f);
    EXPECT_FLOAT_EQ(signedDistanceBox({0.0f, -1.0f, 0.0f}, boxMin, boxMax), 0.0f);
    EXPECT_FLOAT_EQ(signedDistanceBox({1.0f, 1.0f, 0.0f}, boxMin, boxMax), 0.0f);
    EXPECT_FLOAT_EQ(signedDistanceBox({-1.0f, 1.0f, -1.0f}, boxMin, boxMax), 0.0f);

    // Outside box (positive).
    EXPECT_FLOAT_EQ(signedDistanceBox({3.0f, 0.0f, 0.0f}, boxMin, boxMax), 2.0f);
    EXPECT_FLOAT_EQ(signedDistanceBox({4.0f, 5.0f, 0.0f}, boxMin, boxMax), 5.0f);  // 3-4-5 edge distance
    EXPECT_NEAR(signedDistanceBox({2.0f, 3.0f, 7.0f}, boxMin, boxMax), std::sqrt(41.0f), 1e-5f);

    // Voxelized 3D volume preserves exact inside/surface/outside signs and planar distances.
    SdfGridConfig cfg;
    cfg.boundsMin = glm::vec3(-8.0f);
    cfg.boundsMax = glm::vec3(8.0f);
    cfg.resolution = glm::uvec3(64u, 64u, 64u);  // 0.25 m voxels: box faces lie cleanly at cell boundaries

    const SdfBoxProxy proxy{boxMin, boxMax, {0.8f, 0.8f, 0.8f}, {0.0f, 0.0f, 0.0f}};
    const SdfVolumeData volume = buildSdfVolumeFromBoxes(cfg, std::span(&proxy, 1));
    ASSERT_EQ(volume.distanceField.size(), cfg.voxelCount());

    EXPECT_LT(sampleSdfDistance(volume, {0.0f, 0.0f, 0.0f}), -0.75f);
    EXPECT_NEAR(sampleSdfDistance(volume, {0.5f, 0.0f, 0.0f}), -0.5f, 1e-4f);
    EXPECT_NEAR(sampleSdfDistance(volume, {1.0f, 0.0f, 0.0f}), 0.0f, 1e-4f);
    EXPECT_NEAR(sampleSdfDistance(volume, {-1.0f, 0.0f, 0.0f}), 0.0f, 1e-4f);
    EXPECT_NEAR(sampleSdfDistance(volume, {3.0f, 0.0f, 0.0f}), 2.0f, 1e-4f);
    EXPECT_GT(sampleSdfDistance(volume, {0.0f, 0.0f, -5.0f}), 3.9f);
}

TEST(SdfGi, SphereTraceHitsBoxSurfaceAtExpectedDistanceAndPosition) {
    SdfGridConfig cfg;
    cfg.boundsMin = glm::vec3(-8.0f);
    cfg.boundsMax = glm::vec3(8.0f);
    cfg.resolution = glm::uvec3(64u, 64u, 64u);
    cfg.maxSteps = 64u;
    cfg.maxDistance = 32.0f;

    const SdfBoxProxy box{{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}, {0.8f, 0.2f, 0.2f}, {0.0f, 0.0f, 0.0f}};
    SdfVolumeData volume = buildSdfVolumeFromBoxes(cfg, std::span(&box, 1));
    injectDirectLightingToSurfels(volume, glm::normalize(glm::vec3(0.0f, 0.2f, -1.0f)), {3.0f, 3.0f, 3.0f});

    // Trace from z = -5 along +Z -> hits z = -1 at distance 4.0.
    const SdfRayHit hitZ = traceSdfRay(volume, {0.0f, 0.0f, -5.0f}, {0.0f, 0.0f, 1.0f});
    ASSERT_TRUE(hitZ.hit);
    EXPECT_NEAR(hitZ.distance, 4.0f, 0.05f);
    EXPECT_NEAR(hitZ.position.x, 0.0f, 0.05f);
    EXPECT_NEAR(hitZ.position.y, 0.0f, 0.05f);
    EXPECT_NEAR(hitZ.position.z, -1.0f, 0.05f);
    EXPECT_NEAR(hitZ.normal.z, -1.0f, 0.05f);
    EXPECT_GT(hitZ.radiance.r, hitZ.radiance.g);
    EXPECT_GT(hitZ.radiance.r, hitZ.radiance.b);

    // Trace from x = +5 along -X -> hits x = +1 at distance 4.0.
    const SdfRayHit hitX = traceSdfRay(volume, {5.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f});
    ASSERT_TRUE(hitX.hit);
    EXPECT_NEAR(hitX.distance, 4.0f, 0.05f);
    EXPECT_NEAR(hitX.position.x, 1.0f, 0.05f);
    EXPECT_NEAR(hitX.normal.x, 1.0f, 0.05f);

    // Ray passing above the box misses cleanly.
    const SdfRayHit miss = traceSdfRay(volume, {0.0f, 3.5f, -5.0f}, {0.0f, 0.0f, 1.0f});
    EXPECT_FALSE(miss.hit);
}

TEST(SdfGi, CornellBoxRedWallColorBleedingOntoWhiteFloor) {
    const SdfGridConfig cfg = makeTestGridConfig({48u, 32u, 48u});

    const std::vector<SdfBoxProxy> boxes = {
        // White floor
        {{-4.0f, -0.5f, -4.0f}, {4.0f, 0.0f, 4.0f}, {0.85f, 0.85f, 0.85f}, {0.0f, 0.0f, 0.0f}},
        // Sunlit red right wall (x = +2.0)
        {{2.0f, 0.0f, -4.0f}, {2.6f, 4.0f, 4.0f}, {0.92f, 0.04f, 0.04f}, {0.0f, 0.0f, 0.0f}},
        // Green left wall (x = -2.0)
        {{-2.6f, 0.0f, -4.0f}, {-2.0f, 4.0f, 4.0f}, {0.04f, 0.92f, 0.04f}, {0.0f, 0.0f, 0.0f}},
        // White back wall (z = -4.0)
        {{-2.6f, 0.0f, -4.6f}, {2.6f, 4.0f, -4.0f}, {0.80f, 0.80f, 0.80f}, {0.0f, 0.0f, 0.0f}},
    };

    SdfVolumeData volume = buildSdfVolumeFromBoxes(cfg, boxes);
    // Sun shines from (-X, +Y, +Z) directly illuminating the -X face of the red wall and the floor.
    const glm::vec3 sunToRedWall = glm::normalize(glm::vec3(-1.0f, 1.0f, 0.25f));
    injectDirectLightingToSurfels(volume, sunToRedWall, glm::vec3(5.0f, 5.0f, 5.0f));

    const glm::vec3 floorNearRedWall = evaluateSdfGiPoint(volume, {1.25f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    EXPECT_GT(floorNearRedWall.r, 0.05f);
    EXPECT_GT(floorNearRedWall.r, floorNearRedWall.g * 2.0f);
    EXPECT_GT(floorNearRedWall.r, floorNearRedWall.b * 2.0f);

    // Flip sun to illuminate the +X face of the green wall: floor near green wall receives green bounce.
    const glm::vec3 sunToGreenWall = glm::normalize(glm::vec3(1.0f, 1.0f, 0.25f));
    injectDirectLightingToSurfels(volume, sunToGreenWall, glm::vec3(5.0f, 5.0f, 5.0f));

    const glm::vec3 floorNearGreenWall = evaluateSdfGiPoint(volume, {-1.25f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    EXPECT_GT(floorNearGreenWall.g, 0.05f);
    EXPECT_GT(floorNearGreenWall.g, floorNearGreenWall.r * 2.0f);
    EXPECT_GT(floorNearGreenWall.g, floorNearGreenWall.b * 2.0f);
}

TEST(SdfGi, SurfelShadowingReducesOccludedBounce) {
    const SdfGridConfig cfg = makeTestGridConfig({48u, 32u, 48u});
    const std::vector<SdfBoxProxy> unoccluded = {
        {{-4.0f, -0.5f, -4.0f}, {4.0f, 0.0f, 4.0f}, {0.8f, 0.8f, 0.8f}, {0.0f, 0.0f, 0.0f}},
        {{2.0f, 0.0f, -4.0f}, {2.6f, 4.0f, 4.0f}, {0.9f, 0.05f, 0.05f}, {0.0f, 0.0f, 0.0f}},
    };
    std::vector<SdfBoxProxy> occluded = unoccluded;
    // Blocker placed in front of the red wall along the sun vector (-1, 0.5, 0)
    occluded.push_back({{-0.5f, 0.0f, -4.0f}, {0.2f, 4.0f, 4.0f}, {0.1f, 0.1f, 0.1f}, {0.0f, 0.0f, 0.0f}});

    const glm::vec3 sunDir = glm::normalize(glm::vec3(-1.0f, 0.35f, 0.0f));
    SdfVolumeData volOpen = buildSdfVolumeFromBoxes(cfg, unoccluded);
    injectDirectLightingToSurfels(volOpen, sunDir, glm::vec3(5.0f));

    SdfVolumeData volBlocked = buildSdfVolumeFromBoxes(cfg, occluded);
    injectDirectLightingToSurfels(volBlocked, sunDir, glm::vec3(5.0f));

    const glm::vec3 giOpen = evaluateSdfGiPoint(volOpen, {1.2f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    const glm::vec3 giBlocked = evaluateSdfGiPoint(volBlocked, {1.2f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    EXPECT_GT(giOpen.r, giBlocked.r * 3.0f);
}

TEST(SdfGi, BuildSdfVolumeFromSceneSupportsSunAndPointLights) {
    // 1. Directional sun scene
    {
        Scene scene("CornellSun");
        const auto floor = scene.createEntity("WhiteFloor");
        scene.registry().get<TransformComponent>(floor).position = {0.0f, -0.25f, 0.0f};
        scene.registry().get<TransformComponent>(floor).scale = {8.0f, 0.5f, 8.0f};
        auto& floorMesh = scene.registry().emplace<MeshRendererComponent>(floor);
        floorMesh.baseColor = {0.85f, 0.85f, 0.85f};

        const auto wall = scene.createEntity("RedWall");
        scene.registry().get<TransformComponent>(wall).position = {2.3f, 2.0f, 0.0f};
        scene.registry().get<TransformComponent>(wall).scale = {0.6f, 4.0f, 8.0f};
        auto& wallMesh = scene.registry().emplace<MeshRendererComponent>(wall);
        wallMesh.baseColor = {0.9f, 0.05f, 0.05f};

        const auto sun = scene.createEntity("Sun");
        // Euler YXZ: yaw -90 deg rotates -Z to +X; pitch -45 deg tilts downward -> shines toward (+X, -Y)
        scene.registry().get<TransformComponent>(sun).rotationDeg = {-45.0f, -90.0f, 0.0f};
        auto& light = scene.registry().emplace<LightComponent>(sun);
        light.type = LightType::Directional;
        light.color = {1.0f, 1.0f, 1.0f};
        light.intensity = 5.0f;

        scene.updateWorldTransforms();

        const SdfVolumeData volume = buildSdfVolumeFromScene(scene, makeTestGridConfig({48u, 32u, 48u}));
        const glm::vec3 gi = evaluateSdfGiPoint(volume, {1.25f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        EXPECT_GT(gi.r, 0.05f);
        EXPECT_GT(gi.r, gi.g * 2.0f);
        EXPECT_GT(gi.r, gi.b * 2.0f);
    }

    // 2. Interior point light scene
    {
        Scene scene("CornellPointLight");
        const auto floor = scene.createEntity("WhiteFloor");
        scene.registry().get<TransformComponent>(floor).position = {0.0f, -0.25f, 0.0f};
        scene.registry().get<TransformComponent>(floor).scale = {8.0f, 0.5f, 8.0f};
        scene.registry().emplace<MeshRendererComponent>(floor).baseColor = {0.85f, 0.85f, 0.85f};

        const auto wall = scene.createEntity("BlueWall");
        scene.registry().get<TransformComponent>(wall).position = {2.3f, 2.0f, 0.0f};
        scene.registry().get<TransformComponent>(wall).scale = {0.6f, 4.0f, 8.0f};
        scene.registry().emplace<MeshRendererComponent>(wall).baseColor = {0.05f, 0.10f, 0.92f};

        const auto bulb = scene.createEntity("PointBulb");
        scene.registry().get<TransformComponent>(bulb).position = {0.8f, 1.8f, 0.0f};
        auto& pt = scene.registry().emplace<LightComponent>(bulb);
        pt.type = LightType::Point;
        pt.color = {1.0f, 1.0f, 1.0f};
        pt.intensity = 18.0f;
        pt.range = 10.0f;

        scene.updateWorldTransforms();

        const SdfVolumeData volume = buildSdfVolumeFromScene(scene, makeTestGridConfig({48u, 32u, 48u}));
        const glm::vec3 gi = evaluateSdfGiPoint(volume, {1.25f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        EXPECT_GT(gi.b, 0.05f);
        EXPECT_GT(gi.b, gi.r * 2.0f);
        EXPECT_GT(gi.b, gi.g * 2.0f);
    }
}

TEST(SdfGi, MultiBounceAndPushConstantsPacking) {
    SdfGridConfig cfg = makeTestGridConfig({32u, 32u, 32u});
    cfg.boundsMin = {-10.0f, -2.5f, -12.0f};
    cfg.boundsMax = {14.0f, 8.5f, 12.0f};
    cfg.maxSteps = 80u;
    cfg.raysPerPoint = 12u;
    cfg.bounceStrength = 1.25f;
    cfg.skyIntensity = 0.5f;

    const PushConstants pc = SdfGiPass::makePushConstants(11u, 22u, 33u, cfg, 44u, 5u);
    EXPECT_EQ(pc.index0, 11u);
    EXPECT_EQ(pc.index1, 22u);
    EXPECT_EQ(pc.index2, 33u);
    EXPECT_EQ(pc.flags & SdfGiPass::kNoNormalTexture, 44u);
    EXPECT_NE(pc.flags & (1u << 15u), 0u);
    EXPECT_EQ((pc.flags >> 16u) & 0xFFu, 80u);
    EXPECT_EQ(((pc.flags >> 24u) & 0xFu) + 1u, 12u);
    EXPECT_EQ((pc.flags >> 28u) & 0xFu, 5u);

    auto unpack2 = [](float f) {
        const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
        return glm::vec2(glm::unpackHalf1x16(static_cast<glm::uint16>(u & 0xFFFFu)),
                         glm::unpackHalf1x16(static_cast<glm::uint16>(u >> 16u)));
    };
    EXPECT_NEAR(unpack2(pc.param0).x, cfg.boundsMin.x, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param0).y, cfg.boundsMin.y, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param1).x, cfg.boundsMin.z, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param1).y, cfg.boundsMax.x, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param2).x, cfg.boundsMax.y, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param2).y, cfg.boundsMax.z, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param3).x, cfg.bounceStrength, 1e-2f);
    EXPECT_NEAR(unpack2(pc.param3).y, cfg.skyIntensity, 1e-2f);

    const std::vector<SdfBoxProxy> boxes = {
        {{-4.0f, -0.5f, -4.0f}, {4.0f, 0.0f, 4.0f}, {0.8f, 0.8f, 0.8f}, {0.0f, 0.0f, 0.0f}},
        {{2.0f, 0.0f, -4.0f}, {2.6f, 4.0f, 4.0f}, {0.9f, 0.1f, 0.1f}, {0.0f, 0.0f, 0.0f}},
        {{-2.6f, 0.0f, -4.0f}, {-2.0f, 4.0f, 4.0f}, {0.8f, 0.8f, 0.8f}, {0.0f, 0.0f, 0.0f}},
    };
    SdfVolumeData vol = buildSdfVolumeFromBoxes(makeTestGridConfig({32u, 24u, 32u}), boxes);
    injectDirectLightingToSurfels(vol, glm::normalize(glm::vec3(-1.0f, 1.0f, 0.0f)), glm::vec3(4.0f));
    const glm::vec3 giBefore = evaluateSdfGiPoint(vol, {-1.2f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    propagateSdfMultiBounce(vol, 1u, 8u);
    const glm::vec3 giAfter = evaluateSdfGiPoint(vol, {-1.2f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    EXPECT_GE(giAfter.r, giBefore.r);
    EXPECT_TRUE(std::isfinite(giAfter.r) && std::isfinite(giAfter.g) && std::isfinite(giAfter.b));
}

TEST(SdfGiGpu, UploadVolumeCreates3DTexturesAndBindlessIndices) {
    RenderContextConfig config;
    config.requirePresentation = false;
    std::unique_ptr<RenderContext> context;
    try {
        context = std::make_unique<RenderContext>(config);
    } catch (const rhi::UnsupportedHardwareError& e) {
        GTEST_SKIP() << "no suitable GPU: " << e.what();
    }

    const SdfGridConfig cfg = makeTestGridConfig({16u, 16u, 16u});
    const SdfBoxProxy box{{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}, {0.9f, 0.2f, 0.2f}, {0.0f, 0.0f, 0.0f}};
    SdfVolumeData volume = buildSdfVolumeFromBoxes(cfg, std::span(&box, 1));
    injectDirectLightingToSurfels(volume, {0.0f, 1.0f, 0.0f}, {3.0f, 3.0f, 3.0f});

    const bool hasCompiledShader = std::filesystem::exists(paths::shaderDir() / "rtgi.sdf.comp.spv");
    SdfGiPass pass(*context, hasCompiledShader);
    pass.uploadVolume(volume);

    ASSERT_NE(pass.sdfTexture(), nullptr);
    ASSERT_NE(pass.surfelTexture(), nullptr);
    EXPECT_NE(pass.sdfBindlessIndex(), UINT32_MAX);
    EXPECT_NE(pass.surfelBindlessIndex(), UINT32_MAX);
    EXPECT_NE(pass.sdfBindlessIndex(), pass.surfelBindlessIndex());
    EXPECT_TRUE(pass.hasVolume());

    const std::uint32_t firstSdfIdx = pass.sdfBindlessIndex();
    const std::uint32_t firstSurfelIdx = pass.surfelBindlessIndex();
    // Re-uploading at identical resolution updates existing 3D textures in place.
    pass.uploadVolume(volume);
    EXPECT_EQ(pass.sdfBindlessIndex(), firstSdfIdx);
    EXPECT_EQ(pass.surfelBindlessIndex(), firstSurfelIdx);

    if (hasCompiledShader) {
        EXPECT_TRUE(pass.ready());
        RenderGraph graph;
        const RGImage depth = graph.createImage("Depth", {context->depthFormat(), 32u, 32u});
        const RGImage giOut = pass.addPass(graph, depth, 32u, 32u);
        EXPECT_TRUE(giOut.valid());
    }
}

TEST(SdfGiGpu, ComputeShaderProducesRedColorBleedingOnGpu) {
    std::filesystem::path shaderDir = paths::shaderDir();
    if (const char* overrideDir = std::getenv("LUMORA_TEST_SHADER_DIR")) {
        shaderDir = overrideDir;
    }
    if (!std::filesystem::exists(shaderDir / "rtgi.sdf.comp.spv")) {
        GTEST_SKIP() << "rtgi.sdf.comp.spv not found in " << shaderDir.string();
    }

    RenderContextConfig config;
    config.requirePresentation = false;
    config.enableValidation = true;
    config.shaderDirectory = shaderDir;
    std::unique_ptr<RenderContext> context;
    try {
        context = std::make_unique<RenderContext>(config);
    } catch (const rhi::UnsupportedHardwareError& e) {
        GTEST_SKIP() << "no suitable GPU: " << e.what();
    }

    SdfGridConfig cfg = makeTestGridConfig({48u, 32u, 48u});
    cfg.skyIntensity = 0.0f;
    cfg.raysPerPoint = 16u;
    const std::vector<SdfBoxProxy> boxes = {
        {{-4.0f, -0.5f, -4.0f}, {4.0f, 0.0f, 4.0f}, {0.85f, 0.85f, 0.85f}, {0.0f, 0.0f, 0.0f}},
        {{2.0f, 0.0f, -4.0f}, {2.6f, 4.0f, 4.0f}, {0.92f, 0.04f, 0.04f}, {0.0f, 0.0f, 0.0f}},
    };
    SdfVolumeData volume = buildSdfVolumeFromBoxes(cfg, boxes);
    injectDirectLightingToSurfels(volume, glm::normalize(glm::vec3(-1.0f, 1.0f, 0.0f)), glm::vec3(5.0f));

    SdfGiPass pass(*context, true);
    pass.uploadVolume(volume);
    ASSERT_TRUE(pass.ready());

    constexpr std::uint32_t kSize = 16u;
    FlyCameraController cam;
    cam.setPosition({1.0f, 1.4f, 1.8f});
    cam.lookAt({1.2f, 0.0f, 0.0f});
    const CameraView view = cam.view(1.0f);
    const glm::mat4 vp = view.projection * view.view;
    const glm::mat4 invVp = glm::inverse(vp);

    // Synthesize reversed-Z depth buffer of the y=0 floor plane seen by the camera.
    std::vector<float> depthPixels(static_cast<std::size_t>(kSize) * kSize, 0.0f);
    for (std::uint32_t y = 0; y < kSize; ++y) {
        for (std::uint32_t x = 0; x < kSize; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(kSize);
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(kSize);
            const glm::vec2 ndc = glm::vec2(u, v) * 2.0f - 1.0f;
            const glm::vec4 nearW = invVp * glm::vec4(ndc, 1.0f, 1.0f);
            const glm::vec3 pNear = glm::vec3(nearW) / nearW.w;
            const glm::vec3 rayDir = glm::normalize(pNear - view.position);
            if (rayDir.y < -1e-4f) {
                const float t = (0.0f - view.position.y) / rayDir.y;
                const glm::vec3 hitPos = view.position + rayDir * t;
                const glm::vec4 clip = vp * glm::vec4(hitPos, 1.0f);
                depthPixels[static_cast<std::size_t>(y) * kSize + x] = clip.z / clip.w;
            }
        }
    }

    const std::array<std::span<const std::byte>, 1> depthLevels{std::as_bytes(std::span(depthPixels))};
    nvrhi::TextureHandle depthTex = context->createTexture(depthLevels, kSize, kSize, nvrhi::Format::R32_FLOAT, "TestDepth");
    const std::uint32_t depthIdx = context->bindless().registerTexture(depthTex);

    nvrhi::TextureDesc giDesc;
    giDesc.width = kSize;
    giDesc.height = kSize;
    giDesc.format = nvrhi::Format::RGBA16_FLOAT;
    giDesc.isUAV = true;
    giDesc.debugName = "TestSdfGiOut";
    giDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    giDesc.keepInitialState = true;
    nvrhi::TextureHandle giTex = context->gpu()->createTexture(giDesc);
    ASSERT_NE(giTex, nullptr);
    const std::uint32_t giIdx = context->bindless().registerTexture(giTex);

    nvrhi::TextureDesc stageDesc;
    stageDesc.width = kSize;
    stageDesc.height = kSize;
    stageDesc.format = nvrhi::Format::RGBA16_FLOAT;
    stageDesc.debugName = "TestSdfGiStaging";
    nvrhi::StagingTextureHandle staging = context->gpu()->createStagingTexture(stageDesc, nvrhi::CpuAccessMode::Read);
    ASSERT_NE(staging, nullptr);

    FrameConstants fc{};
    fc.view = view.view;
    fc.projection = view.projection;
    fc.viewProjection = vp;
    fc.inverseViewProjection = invVp;
    fc.cameraPositionTime = glm::vec4(view.position, 0.0f);
    fc.viewportSize = glm::vec4(static_cast<float>(kSize), static_cast<float>(kSize),
                                1.0f / static_cast<float>(kSize), 1.0f / static_cast<float>(kSize));
    fc.outputSize = fc.viewportSize;

    FrameResources frame;
    frame.constants.write(context->gpu(), std::as_bytes(std::span(&fc, 1)), DynamicBuffer::Kind::Constant,
                          sizeof(FrameConstants), "TestFrameCB");
    const GpuObject dummyObj{};
    frame.objects.write(context->gpu(), std::as_bytes(std::span(&dummyObj, 1)), DynamicBuffer::Kind::Structured,
                        sizeof(GpuObject), "TestObjects");
    const glm::mat4 dummyJoint(1.0f);
    frame.joints.write(context->gpu(), std::as_bytes(std::span(&dummyJoint, 1)), DynamicBuffer::Kind::Structured,
                       sizeof(glm::mat4), "TestJoints");
    const GpuLight dummyLight{};
    frame.pointLights.write(context->gpu(), std::as_bytes(std::span(&dummyLight, 1)), DynamicBuffer::Kind::Structured,
                            sizeof(GpuLight), "TestLights");
    const std::uint32_t dummyGrid = 0u;
    frame.lightGrid.write(context->gpu(), std::as_bytes(std::span(&dummyGrid, 1)), DynamicBuffer::Kind::Structured,
                          sizeof(std::uint32_t), "TestLightGrid");

    TransientImagePool pool(context->gpu(), context->bindless());
    FramebufferCache fbCache;
    GraphRuntime runtime{};
    runtime.device = context->gpu();
    runtime.pool = &pool;
    runtime.framebuffers = &fbCache;
    runtime.frameBindings = frame.bindingSet(context->gpu(), context->frameLayout(), context->samplers());
    runtime.storageLayout = context->storageLayout();
    runtime.storageFallback = context->storageFallback();

    RenderGraph graph;
    const RGImage rgDepth = graph.importImage("Depth", {depthTex, depthIdx});
    const RGImage rgGi = graph.importImage("GiOut", {giTex, giIdx});
    pass.addDispatchPass(graph, rgGi, rgDepth, kSize, kSize);
    graph.addPass(
        "Readback", [&](PassBuilder& b) { b.copyFrom(rgGi); },
        [&](PassContext& ctx) {
            ctx.cmd->copyTexture(staging, nvrhi::TextureSlice(), ctx.texture(rgGi), nvrhi::TextureSlice());
        });

    nvrhi::CommandListHandle cmd = context->gpu()->createCommandList();
    cmd->open();
    auto lists = graph.execute(cmd, runtime);
    lists.back()->close();
    std::vector<nvrhi::ICommandList*> rawLists;
    rawLists.reserve(lists.size());
    for (const auto& l : lists) rawLists.push_back(l.Get());
    context->gpu()->executeCommandLists(rawLists.data(), rawLists.size());
    context->gpu()->waitForIdle();

    std::size_t rowPitch = 0;
    const auto* mapped = static_cast<const std::uint8_t*>(
        context->gpu()->mapStagingTexture(staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &rowPitch));
    ASSERT_NE(mapped, nullptr);
    const std::size_t centerIdx = static_cast<std::size_t>(kSize / 2u) * rowPitch + static_cast<std::size_t>(kSize / 2u) * 8u;
    const std::uint16_t* rgba16 = reinterpret_cast<const std::uint16_t*>(mapped + centerIdx);
    const float r = glm::unpackHalf1x16(rgba16[0]);
    const float g = glm::unpackHalf1x16(rgba16[1]);
    const float b = glm::unpackHalf1x16(rgba16[2]);
    context->gpu()->unmapStagingTexture(staging);

    context->bindless().releaseTexture(depthIdx);
    context->bindless().releaseTexture(giIdx);

    EXPECT_GT(r, 0.05f);
    EXPECT_GT(r, g * 2.0f);
    EXPECT_GT(r, b * 2.0f);
}
