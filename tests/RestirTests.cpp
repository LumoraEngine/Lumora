#include "renderer/RestirLighting.hpp"
#include "platform/Paths.hpp"
#include "renderer/FrameResources.hpp"
#include "renderer/RenderContext.hpp"
#include "scene/Camera.hpp"

#include <gtest/gtest.h>

#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/trigonometric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

using namespace lumora::render;

namespace {

std::vector<GpuLight> makeTestPointLights(std::uint32_t count, std::uint32_t seed,
                                          bool clusteredBrightLights = false) {
    std::vector<GpuLight> lights;
    lights.reserve(count);
    std::uint32_t rng = wangHash(seed) | 1u;
    for (std::uint32_t i = 0; i < count; ++i) {
        GpuLight light;
        const float u0 = nextRandom01(rng);
        const float u1 = nextRandom01(rng);
        const float u2 = nextRandom01(rng);
        const float u3 = nextRandom01(rng);
        const float u4 = nextRandom01(rng);
        const float u5 = nextRandom01(rng);

        if (clusteredBrightLights) {
            if (i < count / 50u) {
                // Sparse bright lights near the surface.
                light.positionRange = glm::vec4((u0 - 0.5f) * 4.0f, 0.8f + u1 * 1.5f, (u2 - 0.5f) * 4.0f, 12.0f);
                light.colorIntensity = glm::vec4(0.6f + 0.4f * u3, 0.5f + 0.5f * u4, 0.4f + 0.6f * u5, 18.0f + 12.0f * u0);
            } else {
                // Many weak / distant background lights that punish naive 1-sample MC.
                light.positionRange = glm::vec4((u0 - 0.5f) * 28.0f, 0.5f + u1 * 8.0f, (u2 - 0.5f) * 28.0f, 30.0f);
                light.colorIntensity = glm::vec4(0.3f + 0.7f * u3, 0.3f + 0.7f * u4, 0.3f + 0.7f * u5, 0.05f + 0.2f * u1);
            }
        } else {
            light.positionRange = glm::vec4((u0 - 0.5f) * 14.0f, 0.5f + u1 * 5.0f, (u2 - 0.5f) * 14.0f, 18.0f);
            light.colorIntensity = glm::vec4(0.25f + 0.75f * u3, 0.25f + 0.75f * u4, 0.25f + 0.75f * u5, 1.0f + 6.0f * u0);
        }
        light.directionType = glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);  // LIGHT_POINT
        lights.push_back(light);
    }
    return lights;
}

float luminance(const glm::vec3& c) noexcept {
    return glm::dot(c, glm::vec3(0.2126f, 0.7152f, 0.0722f));
}

} // namespace

TEST(RestirLighting, StructLayoutAndDefaultConfigMatchSpec) {
    EXPECT_EQ(sizeof(GpuReservoir), 16u);
    const GpuReservoir r{};
    EXPECT_EQ(r.lightIndex, 0u);
    EXPECT_FLOAT_EQ(r.weightSum, 0.0f);
    EXPECT_EQ(r.sampleCountM, 0u);
    EXPECT_FLOAT_EQ(r.W, 0.0f);

    const RestirConfig cfg{};
    EXPECT_EQ(cfg.initialCandidates, 32u);
    EXPECT_EQ(cfg.spatialNeighbors, 5u);
    EXPECT_FLOAT_EQ(cfg.spatialRadiusPx, 16.0f);
    EXPECT_EQ(cfg.maxTemporalM, 20u);
    EXPECT_FLOAT_EQ(cfg.depthThreshold, 0.05f);
    EXPECT_FLOAT_EQ(cfg.normalThreshold, 0.9f);
}

TEST(RestirLighting, WrsAndRestirMeanMatchesAnalyticalOver2000TrialsAnd500Lights) {
    const auto lights = makeTestPointLights(500u, 1337u, false);

    // 5x5 heterogeneous surface patch (slight normal/position variation across neighbors)
    // to exercise unbiased MIS across differing target PDFs.
    constexpr std::uint32_t kGrid = 5u;
    std::vector<RestirSurfaceSample> surfaces(kGrid * kGrid);
    for (std::uint32_t y = 0; y < kGrid; ++y) {
        for (std::uint32_t x = 0; x < kGrid; ++x) {
            RestirSurfaceSample s;
            const float dx = static_cast<float>(x) - 2.0f;
            const float dy = static_cast<float>(y) - 2.0f;
            s.position = glm::vec3(dx * 0.15f, 0.0f, dy * 0.15f);
            s.normal = glm::normalize(glm::vec3(dx * 0.06f, 1.0f, dy * 0.06f));
            s.albedo = glm::vec3(0.8f, 0.75f, 0.7f);
            s.depth = 5.0f + dy * 0.02f;
            s.valid = true;
            surfaces[y * kGrid + x] = s;
        }
    }

    const std::size_t centerIdx = 2u * kGrid + 2u;
    const RestirSurfaceSample& centerSurface = surfaces[centerIdx];
    const glm::vec3 analytical = evaluateDirectLightingCpu(lights, centerSurface);
    ASSERT_GT(luminance(analytical), 0.1f);

    RestirConfig cfg;
    cfg.initialCandidates = 32u;
    cfg.spatialNeighbors = 5u;
    cfg.spatialRadiusPx = 3.0f;
    cfg.maxTemporalM = 20u;
    cfg.depthThreshold = 0.05f;
    cfg.normalThreshold = 0.9f;

    constexpr std::uint32_t kTrials = 2000u;
    glm::dvec3 wrsSum(0.0);
    glm::dvec3 restirSum(0.0);

    std::uint32_t wrsRng = wangHash(42u) | 1u;
    const float sourcePdf = 1.0f / static_cast<float>(lights.size());

    std::vector<GpuReservoir> prevRes;
    std::vector<GpuReservoir> currRes;
    std::vector<glm::vec3> hdrOut;

    for (std::uint32_t t = 0; t < kTrials; ++t) {
        // 1. Pure WRS trial at center pixel
        GpuReservoir wrs{};
        for (std::uint32_t c = 0; c < cfg.initialCandidates; ++c) {
            const std::uint32_t idx =
                std::min(static_cast<std::uint32_t>(nextRandom01(wrsRng) * static_cast<float>(lights.size())),
                         static_cast<std::uint32_t>(lights.size()) - 1u);
            const float pHat = evaluateTargetPdf(lights[idx], centerSurface);
            updateReservoir(wrs, idx, pHat / sourcePdf, nextRandom01(wrsRng), 1u);
        }
        finalizeReservoirWeight(wrs, evaluateTargetPdf(lights[wrs.lightIndex], centerSurface));
        wrsSum += glm::dvec3(evaluateLightContribution(lights[wrs.lightIndex], centerSurface) * wrs.W);

        // 2. Full 2-frame spatiotemporal ReSTIR DI trial (independent frame pair per trial)
        evaluateRestirFrameCpu(kGrid, kGrid, surfaces, lights, cfg, t * 2u, prevRes, hdrOut);
        evaluateRestirFrameCpu(kGrid, kGrid, surfaces, lights, cfg, t * 2u + 1u, currRes, hdrOut, prevRes, surfaces);
        restirSum += glm::dvec3(hdrOut[centerIdx]);
    }

    const glm::vec3 wrsMean = glm::vec3(wrsSum / static_cast<double>(kTrials));
    const glm::vec3 restirMean = glm::vec3(restirSum / static_cast<double>(kTrials));

    const float analyticalLum = luminance(analytical);
    EXPECT_NEAR(wrsMean.r, analytical.r, analytical.r * 0.04f + 1e-3f);
    EXPECT_NEAR(wrsMean.g, analytical.g, analytical.g * 0.04f + 1e-3f);
    EXPECT_NEAR(wrsMean.b, analytical.b, analytical.b * 0.04f + 1e-3f);
    EXPECT_NEAR(luminance(wrsMean), analyticalLum, analyticalLum * 0.03f);

    EXPECT_NEAR(restirMean.r, analytical.r, analytical.r * 0.04f + 1e-3f);
    EXPECT_NEAR(restirMean.g, analytical.g, analytical.g * 0.04f + 1e-3f);
    EXPECT_NEAR(restirMean.b, analytical.b, analytical.b * 0.04f + 1e-3f);
    EXPECT_NEAR(luminance(restirMean), analyticalLum, analyticalLum * 0.03f);
}

TEST(RestirLighting, TemporalSampleCountMClampingHoldsOver100Frames) {
    const auto lights = makeTestPointLights(500u, 9001u, false);

    constexpr std::uint32_t kWidth = 8u;
    constexpr std::uint32_t kHeight = 8u;
    std::vector<RestirSurfaceSample> surfaces(kWidth * kHeight);
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            RestirSurfaceSample s;
            s.position = glm::vec3(static_cast<float>(x) * 0.25f - 1.0f, 0.0f, static_cast<float>(y) * 0.25f - 1.0f);
            s.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            s.albedo = glm::vec3(1.0f);
            s.depth = 4.0f;
            s.valid = true;
            surfaces[y * kWidth + x] = s;
        }
    }

    RestirConfig cfg;
    cfg.initialCandidates = 32u;
    cfg.maxTemporalM = 20u;
    cfg.spatialNeighbors = 5u;
    cfg.spatialRadiusPx = 4.0f;

    std::vector<GpuReservoir> prevRes;
    std::vector<GpuReservoir> currRes;
    std::vector<glm::vec3> hdrOut;

    for (std::uint32_t frame = 0; frame < 100u; ++frame) {
        evaluateRestirFrameCpu(kWidth, kHeight, surfaces, lights, cfg, frame, currRes, hdrOut, prevRes, surfaces);
        ASSERT_EQ(currRes.size(), kWidth * kHeight);
        for (std::size_t i = 0; i < currRes.size(); ++i) {
            const GpuReservoir& r = currRes[i];
            EXPECT_TRUE(std::isfinite(r.weightSum));
            EXPECT_TRUE(std::isfinite(r.W));
            EXPECT_GE(r.W, 0.0f);
            EXPECT_LT(r.lightIndex, lights.size());
            if (frame == 0u) {
                EXPECT_EQ(r.sampleCountM, cfg.initialCandidates);
            } else {
                // prev.M is clamped <= maxTemporalM before combining with initialCandidates.
                EXPECT_LE(r.sampleCountM, cfg.initialCandidates + cfg.maxTemporalM);
                EXPECT_EQ(r.sampleCountM, cfg.initialCandidates + cfg.maxTemporalM);
            }
        }
        prevRes = currRes;
    }

    // Direct pairwise combineReservoirs clamping check over 100 iterations.
    GpuReservoir prev{};
    prev.lightIndex = 0u;
    prev.sampleCountM = 999u;
    prev.W = 1.0f;
    std::uint32_t rng = wangHash(777u) | 1u;
    for (std::uint32_t frame = 0; frame < 100u; ++frame) {
        GpuReservoir curr{};
        const float p0 = evaluateTargetPdf(lights[0], surfaces[0]);
        updateReservoir(curr, 0u, p0 * static_cast<float>(lights.size()), nextRandom01(rng), cfg.initialCandidates);
        finalizeReservoirWeight(curr, p0);

        combineReservoirs(curr, prev, p0, nextRandom01(rng), cfg.maxTemporalM);
        EXPECT_LE(curr.sampleCountM, cfg.initialCandidates + cfg.maxTemporalM);
        finalizeReservoirWeight(curr, p0);
        prev = curr;
    }
}

TEST(RestirLighting, SpatioTemporalVarianceIsSignificantlyLowerThanNaive1SampleMc) {
    const auto lights = makeTestPointLights(1000u, 2026u, true);

    constexpr std::uint32_t kWidth = 16u;
    constexpr std::uint32_t kHeight = 16u;
    std::vector<RestirSurfaceSample> surfaces(kWidth * kHeight);
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            RestirSurfaceSample s;
            s.position = glm::vec3((static_cast<float>(x) - 7.5f) * 0.2f, 0.0f,
                                   (static_cast<float>(y) - 7.5f) * 0.2f);
            s.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            s.albedo = glm::vec3(0.9f, 0.9f, 0.9f);
            s.depth = 6.0f;
            s.valid = true;
            surfaces[y * kWidth + x] = s;
        }
    }

    RestirConfig cfg;
    cfg.initialCandidates = 32u;
    cfg.spatialNeighbors = 5u;
    cfg.spatialRadiusPx = 6.0f;
    cfg.maxTemporalM = 20u;

    std::vector<GpuReservoir> prevRes;
    std::vector<GpuReservoir> currRes;
    std::vector<glm::vec3> restirHdr;

    // Warm up temporal history for 8 frames.
    for (std::uint32_t f = 0; f < 8u; ++f) {
        evaluateRestirFrameCpu(kWidth, kHeight, surfaces, lights, cfg, f, currRes, restirHdr, prevRes, surfaces);
        prevRes = currRes;
    }

    // Compute analytical ground truth and 1-sample uniform MC across the 16x16 image.
    double naiveMse = 0.0;
    double restirMse = 0.0;
    std::uint32_t mcRng = wangHash(5555u) | 1u;
    const float invPdf = static_cast<float>(lights.size());

    for (std::size_t i = 0; i < surfaces.size(); ++i) {
        const glm::vec3 truth = evaluateDirectLightingCpu(lights, surfaces[i]);
        const float truthLum = luminance(truth);

        const std::uint32_t pick =
            std::min(static_cast<std::uint32_t>(nextRandom01(mcRng) * invPdf),
                     static_cast<std::uint32_t>(lights.size()) - 1u);
        const glm::vec3 mcEstimate = evaluateLightContribution(lights[pick], surfaces[i]) * invPdf;
        const float mcErr = luminance(mcEstimate) - truthLum;
        naiveMse += static_cast<double>(mcErr) * static_cast<double>(mcErr);

        const float restirErr = luminance(restirHdr[i]) - truthLum;
        restirMse += static_cast<double>(restirErr) * static_cast<double>(restirErr);
    }

    naiveMse /= static_cast<double>(surfaces.size());
    restirMse /= static_cast<double>(surfaces.size());

    EXPECT_GT(naiveMse, 0.0);
    EXPECT_LT(restirMse * 25.0, naiveMse)
        << "Expected ReSTIR DI variance (" << restirMse << ") to be >25x smaller than 1-sample MC (" << naiveMse << ")";
}

TEST(RestirLighting, EdgeStoppingAndMotionReprojectionWorkCorrectly) {
    const auto lights = makeTestPointLights(64u, 77u, false);
    RestirConfig cfg;
    cfg.initialCandidates = 16u;
    cfg.maxTemporalM = 20u;
    cfg.spatialNeighbors = 0u;
    cfg.depthThreshold = 0.05f;
    cfg.normalThreshold = 0.9f;

    RestirSurfaceSample sA;
    sA.position = {0.0f, 0.0f, 0.0f};
    sA.normal = {0.0f, 1.0f, 0.0f};
    sA.depth = 10.0f;

    RestirSurfaceSample sDepthJump = sA;
    sDepthJump.depth = 12.0f;  // 20% depth jump > 5% threshold
    EXPECT_FALSE(isSurfaceCompatible(sA, sDepthJump, cfg));

    RestirSurfaceSample sNormalTurn = sA;
    sNormalTurn.normal = glm::normalize(glm::vec3(1.0f, 1.0f, 0.0f));  // dot ~ 0.707 < 0.9
    EXPECT_FALSE(isSurfaceCompatible(sA, sNormalTurn, cfg));

    // Motion reprojection test across a 4x1 strip: pixel 2 has motion moving from pixel 1.
    std::vector<RestirSurfaceSample> prevSurfaces(4, sA);
    std::vector<RestirSurfaceSample> currSurfaces(4, sA);
    currSurfaces[2].motion = glm::vec2(0.5f, 0.0f);  // previousUv = uv - 0.25 -> shifts from x=2 (0.625) to x=1 (0.375)
    // Make pixel 2 compatible with pixel 1, while pixel 3 has incompatible depth (rejected!).
    currSurfaces[3].motion = glm::vec2(0.0f, 0.0f);
    prevSurfaces[3].depth = 20.0f;

    std::vector<GpuReservoir> prevRes;
    std::vector<GpuReservoir> currRes;
    std::vector<glm::vec3> hdrOut;
    evaluateRestirFrameCpu(4u, 1u, prevSurfaces, lights, cfg, 0u, prevRes, hdrOut);
    evaluateRestirFrameCpu(4u, 1u, currSurfaces, lights, cfg, 1u, currRes, hdrOut, prevRes, prevSurfaces);

    EXPECT_EQ(currRes[2].sampleCountM, cfg.initialCandidates + cfg.initialCandidates);  // 16 + min(16, 20) = 32
    EXPECT_EQ(currRes[3].sampleCountM, cfg.initialCandidates);  // history rejected due to depth discontinuity
}

TEST(RestirLighting, SpotLightConeAndEdgeCasesAreSafe) {
    GpuLight spot;
    spot.positionRange = glm::vec4(0.0f, 4.0f, 0.0f, 20.0f);
    spot.colorIntensity = glm::vec4(1.0f, 0.9f, 0.8f, 10.0f);
    spot.directionType = glm::vec4(0.0f, -1.0f, 0.0f, 2.0f);  // LIGHT_SPOT pointing straight down
    spot.spot = glm::vec4(std::cos(glm::radians(30.0f)), std::cos(glm::radians(15.0f)), -1.0f, 0.0f);

    RestirSurfaceSample inside;
    inside.position = {0.0f, 0.0f, 0.0f};
    inside.normal = {0.0f, 1.0f, 0.0f};
    EXPECT_GT(evaluateTargetPdf(spot, inside), 0.0f);

    RestirSurfaceSample outside = inside;
    outside.position = {10.0f, 0.0f, 0.0f};  // outside 30 deg half-angle
    EXPECT_FLOAT_EQ(evaluateTargetPdf(spot, outside), 0.0f);

    // NaN / negative weight safety
    GpuReservoir r{};
    EXPECT_FALSE(updateReservoir(r, 0u, -5.0f, 0.0f, 1u));
    EXPECT_FALSE(updateReservoir(r, 0u, std::numeric_limits<float>::quiet_NaN(), 0.0f, 1u));
    EXPECT_EQ(r.sampleCountM, 2u);
    EXPECT_FLOAT_EQ(r.weightSum, 0.0f);
    finalizeReservoirWeight(r, 1.0f);
    EXPECT_FLOAT_EQ(r.W, 0.0f);

    // Empty light list
    std::vector<RestirSurfaceSample> one{inside};
    std::vector<GpuReservoir> outR;
    std::vector<glm::vec3> outHdr;
    evaluateRestirFrameCpu(1u, 1u, one, {}, RestirConfig{}, 0u, outR, outHdr);
    ASSERT_EQ(outR.size(), 1u);
    EXPECT_FLOAT_EQ(outHdr[0].x, 0.0f);
    EXPECT_FLOAT_EQ(outR[0].W, 0.0f);
}

TEST(RestirLighting, GpuPassDispatchesCleanlyWhenShadersAndVulkanAvailable) {
    std::filesystem::path shaderDir;
    if (const char* envDir = std::getenv("LUMORA_RESTIR_TEST_SHADER_DIR");
        envDir != nullptr && *envDir != '\0') {
        shaderDir = envDir;
    } else if (const char* testDir = std::getenv("LUMORA_TEST_SHADER_DIR");
               testDir != nullptr && *testDir != '\0') {
        shaderDir = testDir;
    } else {
        shaderDir = lumora::paths::shaderDir();
    }
    if (!std::filesystem::exists(shaderDir / "restir.initial.comp.spv") ||
        !std::filesystem::exists(shaderDir / "restir.spatial.comp.spv")) {
        GTEST_SKIP() << "restir SPIR-V shaders not present in " << shaderDir;
    }

    std::unique_ptr<RenderContext> ctx;
    try {
        RenderContextConfig cfg;
        cfg.requirePresentation = false;
        cfg.enableValidation = std::getenv("LUMORA_TEST_VALIDATION") != nullptr;
        cfg.shaderDirectory = shaderDir;
        ctx = std::make_unique<RenderContext>(cfg);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Vulkan context unavailable: " << e.what();
    }

    RestirPass pass(*ctx);
    ASSERT_NE(pass.initialPipeline(), nullptr);
    ASSERT_NE(pass.spatialPipeline(), nullptr);

    constexpr std::uint32_t kW = 16u;
    constexpr std::uint32_t kH = 16u;
    pass.ensureResources(kW, kH);
    ASSERT_NE(pass.hdrOutput(), nullptr);

    // Create minimal depth, normal, motion textures.
    std::vector<float> depthData(kW * kH, 0.01f);
    std::vector<float> normalData(kW * kH * 4u, 0.0f);
    for (std::size_t i = 0; i < kW * kH; ++i) {
        normalData[i * 4u + 1u] = 1.0f;  // +Y normal
        normalData[i * 4u + 3u] = 0.5f;  // roughness
    }
    std::vector<float> motionData(kW * kH * 2u, 0.0f);

    const std::array<std::span<const std::byte>, 1> depthLvl{std::as_bytes(std::span(depthData))};
    const std::array<std::span<const std::byte>, 1> normLvl{std::as_bytes(std::span(normalData))};
    const std::array<std::span<const std::byte>, 1> motLvl{std::as_bytes(std::span(motionData))};

    auto depthTex = ctx->createTexture(depthLvl, kW, kH, nvrhi::Format::R32_FLOAT, "TestDepth");
    auto normTex = ctx->createTexture(normLvl, kW, kH, nvrhi::Format::RGBA32_FLOAT, "TestNormal");
    auto motTex = ctx->createTexture(motLvl, kW, kH, nvrhi::Format::RG32_FLOAT, "TestMotion");

    const std::uint32_t depthIdx = ctx->bindless().registerTexture(depthTex);
    const std::uint32_t normIdx = ctx->bindless().registerTexture(normTex);
    const std::uint32_t motIdx = ctx->bindless().registerTexture(motTex);

    FrameResources frame;
    FrameConstants fc{};
    lumora::FlyCameraController cam;
    cam.setPosition({0.0f, 3.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const lumora::CameraView cv = cam.view(1.0f);
    fc.view = cv.view;
    fc.projection = cv.projection;
    fc.viewProjection = cv.projection * cv.view;
    fc.inverseViewProjection = glm::inverse(fc.viewProjection);
    fc.previousViewProjection = fc.viewProjection;
    fc.cameraPositionTime = glm::vec4(cv.position, 0.0f);
    fc.viewportSize = glm::vec4(static_cast<float>(kW), static_cast<float>(kH), 1.0f / kW, 1.0f / kH);
    fc.outputSize = fc.viewportSize;

    const auto lights = makeTestPointLights(64u, 99u, false);
    fc.lightGrid = glm::uvec4(1u, 1u, 32u, static_cast<std::uint32_t>(lights.size()));

    nvrhi::IDevice* gpu = ctx->gpu();
    frame.constants.write(gpu, std::as_bytes(std::span(&fc, 1)), DynamicBuffer::Kind::Constant,
                          sizeof(FrameConstants), "TestFC");
    const std::array<GpuObject, 1> dummyObj{};
    frame.objects.write(gpu, std::as_bytes(std::span(dummyObj)), DynamicBuffer::Kind::Structured,
                        sizeof(GpuObject), "TestObj");
    const std::array<glm::mat4, 1> dummyJoint{glm::mat4(1.0f)};
    frame.joints.write(gpu, std::as_bytes(std::span(dummyJoint)), DynamicBuffer::Kind::Structured,
                       sizeof(glm::mat4), "TestJoints");
    frame.pointLights.write(gpu, std::as_bytes(std::span(lights)), DynamicBuffer::Kind::Structured,
                            sizeof(GpuLight), "TestLights");
    const std::array<std::uint32_t, 1> dummyGrid{0u};
    frame.lightGrid.write(gpu, std::as_bytes(std::span(dummyGrid)), DynamicBuffer::Kind::Structured,
                          sizeof(std::uint32_t), "TestGrid");
    nvrhi::IBindingSet* frameSet = frame.bindingSet(gpu, ctx->frameLayout(), ctx->samplers());

    RestirDispatchParams dp;
    dp.width = kW;
    dp.height = kH;
    dp.depthBindlessIndex = depthIdx;
    dp.motionBindlessIndex = motIdx;
    dp.normalBindlessIndex = normIdx;

    nvrhi::TextureDesc readDesc;
    readDesc.width = kW;
    readDesc.height = kH;
    readDesc.format = nvrhi::Format::RGBA16_FLOAT;
    readDesc.debugName = "RestirReadback";
    auto staging = gpu->createStagingTexture(readDesc, nvrhi::CpuAccessMode::Read);
    ASSERT_NE(staging, nullptr);

    nvrhi::TextureDesc resReadDesc;
    resReadDesc.width = kW;
    resReadDesc.height = kH;
    resReadDesc.format = nvrhi::Format::RGBA32_UINT;
    resReadDesc.debugName = "RestirResReadback";
    auto resStaging = gpu->createStagingTexture(resReadDesc, nvrhi::CpuAccessMode::Read);
    ASSERT_NE(resStaging, nullptr);

    auto cmd = gpu->createCommandList();
    cmd->open();
    pass.dispatch(cmd, frameSet, dp);
    pass.dispatch(cmd, frameSet, dp);
    cmd->copyTexture(staging, nvrhi::TextureSlice(), pass.hdrOutput(), nvrhi::TextureSlice());
    cmd->copyTexture(resStaging, nvrhi::TextureSlice(), pass.currentReservoirTexture(), nvrhi::TextureSlice());
    cmd->close();
    nvrhi::ICommandList* list = cmd;
    gpu->executeCommandLists(&list, 1);
    gpu->waitForIdle();

    EXPECT_EQ(pass.frameIndex(), 2u);
    EXPECT_TRUE(pass.historyValid());

    std::size_t resPitch = 0;
    const auto* resMapped = static_cast<const std::uint8_t*>(
        gpu->mapStagingTexture(resStaging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &resPitch));
    ASSERT_NE(resMapped, nullptr);
    const auto* resRow0 = reinterpret_cast<const std::uint32_t*>(resMapped);
    GpuReservoir r0{};
    r0.lightIndex = resRow0[0];
    std::memcpy(&r0.weightSum, &resRow0[1], sizeof(float));
    r0.sampleCountM = resRow0[2];
    std::memcpy(&r0.W, &resRow0[3], sizeof(float));
    gpu->unmapStagingTexture(resStaging);

    std::size_t pitch = 0;
    const auto* mapped = static_cast<const std::uint8_t*>(
        gpu->mapStagingTexture(staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &pitch));
    ASSERT_NE(mapped, nullptr);
    double totalGpuLum = 0.0;
    float firstAlpha = -1.0f;
    for (std::uint32_t y = 0; y < kH; ++y) {
        const auto* row = reinterpret_cast<const std::uint16_t*>(mapped + y * pitch);
        for (std::uint32_t x = 0; x < kW; ++x) {
            const float r = glm::unpackHalf1x16(row[x * 4u + 0u]);
            const float g = glm::unpackHalf1x16(row[x * 4u + 1u]);
            const float b = glm::unpackHalf1x16(row[x * 4u + 2u]);
            const float a = glm::unpackHalf1x16(row[x * 4u + 3u]);
            if (x == 0u && y == 0u) {
                firstAlpha = a;
            }
            EXPECT_TRUE(std::isfinite(r));
            EXPECT_TRUE(std::isfinite(g));
            EXPECT_TRUE(std::isfinite(b));
            EXPECT_GE(r, 0.0f);
            EXPECT_GE(g, 0.0f);
            EXPECT_GE(b, 0.0f);
            totalGpuLum += static_cast<double>(luminance({r, g, b}));
        }
    }
    gpu->unmapStagingTexture(staging);
    EXPECT_GT(totalGpuLum, 0.0) << "firstAlpha=" << firstAlpha << " r0=(idx=" << r0.lightIndex
                                << ", wSum=" << r0.weightSum << ", M=" << r0.sampleCountM << ", W=" << r0.W << ")"
                                << " depthIdx=" << depthIdx << " normIdx=" << normIdx << " motIdx=" << motIdx;

    ctx->bindless().releaseTexture(depthIdx);
    ctx->bindless().releaseTexture(normIdx);
    ctx->bindless().releaseTexture(motIdx);
}
