#include "renderer/GpuCulling.hpp"

#include "platform/Paths.hpp"
#include "renderer/FrameResources.hpp"
#include "renderer/GpuTypes.hpp"
#include "renderer/RenderContext.hpp"
#include "scene/Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

using namespace lumora;
using namespace lumora::render;

namespace {

/// Builds a finite reversed-Z perspective projection matrix (Vulkan Y-down, near=1, far=0)
/// to test both the infinite and finite far-plane cases.
glm::mat4 finitePerspectiveReverseZ(float fovYRadians, float aspect, float nearPlane, float farPlane) {
    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    glm::mat4 m(0.0f);
    m[0][0] = f / aspect;
    m[1][1] = -f;
    m[2][2] = nearPlane / (farPlane - nearPlane);
    m[2][3] = -1.0f;
    m[3][2] = (farPlane * nearPlane) / (farPlane - nearPlane);
    return m;
}

/// Projects a world-space quad [minX..maxX] x [minY..maxY] at fixed world Z into a CPU
/// reversed-Z depth buffer (near=1, far=0) matching Lumora's NDC -> UV convention.
void rasterizeWallToDepthBuffer(std::vector<float>& depth,
                                std::uint32_t width,
                                std::uint32_t height,
                                const glm::mat4& vp,
                                glm::vec3 wallMin,
                                glm::vec3 wallMax) {
    // Evaluate all 8 corners of the wall AABB to determine its conservative screen rect and
    // minimum (farthest) reversed-Z depth across the wall face.
    glm::vec2 uvMin(1.0f, 1.0f);
    glm::vec2 uvMax(0.0f, 0.0f);
    float wallMinZ = 1.0f;
    for (int c = 0; c < 8; ++c) {
        const glm::vec3 p(
            (c & 1) != 0 ? wallMax.x : wallMin.x,
            (c & 2) != 0 ? wallMax.y : wallMin.y,
            (c & 4) != 0 ? wallMax.z : wallMin.z);
        const glm::vec4 clip = vp * glm::vec4(p, 1.0f);
        if (clip.w <= 0.0f) continue;
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        const glm::vec2 uv = glm::vec2(ndc.x, ndc.y) * 0.5f + 0.5f;
        uvMin = glm::min(uvMin, uv);
        uvMax = glm::max(uvMax, uv);
        wallMinZ = std::min(wallMinZ, std::clamp(ndc.z, 0.0f, 1.0f));
    }
    uvMin = glm::clamp(uvMin, glm::vec2(0.0f), glm::vec2(1.0f));
    uvMax = glm::clamp(uvMax, glm::vec2(0.0f), glm::vec2(1.0f));

    const int x0 = std::clamp(static_cast<int>(std::floor(uvMin.x * static_cast<float>(width))), 0, static_cast<int>(width) - 1);
    const int x1 = std::clamp(static_cast<int>(std::floor(uvMax.x * static_cast<float>(width))), 0, static_cast<int>(width) - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(uvMin.y * static_cast<float>(height))), 0, static_cast<int>(height) - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(uvMax.y * static_cast<float>(height))), 0, static_cast<int>(height) - 1);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            float& px = depth[static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x)];
            px = std::max(px, wallMinZ);
        }
    }
}

CullInstance makeInstance(glm::vec3 center, glm::vec3 extents, std::uint32_t objectIndex,
                          std::uint32_t indexCount = 36u, std::uint32_t firstIndex = 0u,
                          std::int32_t vertexOffset = 0, std::uint32_t flags = 0u) {
    CullInstance inst;
    inst.aabbCenter = center;
    inst.indexCount = indexCount;
    inst.aabbExtents = extents;
    inst.firstIndex = firstIndex;
    inst.vertexOffset = vertexOffset;
    inst.objectIndex = objectIndex;
    inst.flags = flags;
    return inst;
}

template <typename T>
std::vector<T> readBufferVec(RenderContext& ctx, nvrhi::IBuffer* src, std::size_t count) {
    nvrhi::IDevice* gpu = ctx.gpu();
    nvrhi::BufferDesc rd;
    rd.byteSize = std::max<std::uint64_t>(count * sizeof(T), 4ull);
    rd.cpuAccess = nvrhi::CpuAccessMode::Read;
    rd.debugName = "Readback";
    nvrhi::BufferHandle staging = gpu->createBuffer(rd);
    ctx.upload([&](nvrhi::ICommandList* cmd) {
        cmd->copyBuffer(staging, 0, src, 0, count * sizeof(T));
    });
    gpu->waitForIdle();
    const void* mapped = gpu->mapBuffer(staging, nvrhi::CpuAccessMode::Read);
    std::vector<T> out(count);
    std::memcpy(out.data(), mapped, count * sizeof(T));
    gpu->unmapBuffer(staging);
    return out;
}

} // namespace

TEST(GpuCulling, ReversedZHiZMinReductionEvenDimensions) {
    // 4x4 reversed-Z depth buffer: mostly close occluders (0.85), with one sky gap (0.0) in the
    // top-left 2x2 block and a shallower depth (0.35) in the bottom-right 2x2 block.
    std::vector<float> mip0(16, 0.85f);
    mip0[1 * 4 + 1] = 0.0f;   // (1, 1) -> mip1(0, 0)
    mip0[0 * 4 + 3] = 0.55f;  // (3, 0) -> mip1(1, 0)
    mip0[3 * 4 + 0] = 0.45f;  // (0, 3) -> mip1(0, 1)
    mip0[2 * 4 + 2] = 0.35f;  // (2, 2) -> mip1(1, 1)

    const auto mips = buildHiZMipsCpu(mip0, 4, 4);
    ASSERT_EQ(mips.size(), 3u);
    EXPECT_EQ(mips[0].width, 4u);
    EXPECT_EQ(mips[0].height, 4u);
    EXPECT_EQ(mips[1].width, 2u);
    EXPECT_EQ(mips[1].height, 2u);
    EXPECT_EQ(mips[2].width, 1u);
    EXPECT_EQ(mips[2].height, 1u);

    EXPECT_FLOAT_EQ(mips[1].depths[0], 0.0f);   // min(0.85, 0.85, 0.85, 0.0)
    EXPECT_FLOAT_EQ(mips[1].depths[1], 0.55f);  // min(0.85, 0.55, 0.85, 0.85)
    EXPECT_FLOAT_EQ(mips[1].depths[2], 0.45f);  // min(0.85, 0.85, 0.45, 0.85)
    EXPECT_FLOAT_EQ(mips[1].depths[3], 0.35f);  // min(0.35, 0.85, 0.85, 0.85)
    EXPECT_FLOAT_EQ(mips[2].depths[0], 0.0f);   // global minimum
}

TEST(GpuCulling, ReversedZHiZMinReductionOddDimensionsConservative) {
    // 5x5 -> 2x2 -> 1x1: both width and height are odd.
    // 1) A far texel at the extreme odd boundary (4, 4) MUST be captured by mip1(1, 1).
    {
        std::vector<float> mip0(25, 0.9f);
        mip0[4 * 5 + 4] = 0.05f;
        const auto mips = buildHiZMipsCpu(mip0, 5, 5);
        ASSERT_EQ(mips.size(), 3u);
        EXPECT_EQ(mips[1].width, 2u);
        EXPECT_EQ(mips[1].height, 2u);
        EXPECT_FLOAT_EQ(mips[1].depths[0], 0.9f);
        EXPECT_FLOAT_EQ(mips[1].depths[1], 0.9f);
        EXPECT_FLOAT_EQ(mips[1].depths[2], 0.9f);
        EXPECT_FLOAT_EQ(mips[1].depths[3], 0.05f);
        EXPECT_FLOAT_EQ(mips[2].depths[0], 0.05f);
    }

    // 2) A sky gap at middle column x=2, y=0 overlaps UV [0.4, 0.6], which straddles BOTH
    //    mip1(0, 0) [0.0, 0.5] and mip1(1, 0) [0.5, 1.0] -> both MUST take the minimum 0.0.
    {
        std::vector<float> mip0(25, 0.9f);
        mip0[0 * 5 + 2] = 0.0f;
        const auto mips = buildHiZMipsCpu(mip0, 5, 5);
        ASSERT_EQ(mips.size(), 3u);
        EXPECT_FLOAT_EQ(mips[1].depths[0], 0.0f);
        EXPECT_FLOAT_EQ(mips[1].depths[1], 0.0f);
        EXPECT_FLOAT_EQ(mips[1].depths[2], 0.9f);
        EXPECT_FLOAT_EQ(mips[1].depths[3], 0.9f);
    }

    // 3) Rectangular odd dimensions 7x3 -> 3x1 -> 1x1.
    {
        std::vector<float> mip0(21, 0.8f);
        mip0[2 * 7 + 6] = 0.12f;  // bottom-right odd boundary (6, 2)
        const auto mips = buildHiZMipsCpu(mip0, 7, 3);
        ASSERT_EQ(mips.size(), 3u);
        EXPECT_EQ(mips[1].width, 3u);
        EXPECT_EQ(mips[1].height, 1u);
        EXPECT_EQ(mips[2].width, 1u);
        EXPECT_EQ(mips[2].height, 1u);
        EXPECT_FLOAT_EQ(mips[1].depths[2], 0.12f);
        EXPECT_FLOAT_EQ(mips[2].depths[0], 0.12f);
    }
}

TEST(GpuCulling, ReversedZFrustumPlanesExtractAndCull) {
    FlyCameraController cam;
    cam.fovDeg = 60.0f;
    cam.nearPlane = 0.1f;
    cam.setPosition({0.0f, 0.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const CameraView view = cam.view(1.0f);
    const glm::mat4 vp = view.projection * view.view;
    const auto planes = extractReversedZFrustumPlanes(vp);

    // Planes should be normalized (except the infinite far plane whose normal is zero).
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(glm::length(glm::vec3(planes[i])), 1.0f, 1e-5f);
    }

    const std::vector<CullInstance> instances = {
        makeInstance({0.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f}, 0u),     // center: visible
        makeInstance({0.0f, 0.0f, 10.0f}, {0.5f, 0.5f, 0.5f}, 1u),    // behind camera: culled
        makeInstance({-30.0f, 0.0f, -2.0f}, {0.5f, 0.5f, 0.5f}, 2u),  // left of frustum: culled
        makeInstance({30.0f, 0.0f, -2.0f}, {0.5f, 0.5f, 0.5f}, 3u),   // right of frustum: culled
        makeInstance({0.0f, 30.0f, -2.0f}, {0.5f, 0.5f, 0.5f}, 4u),   // above frustum: culled
        makeInstance({0.0f, -30.0f, -2.0f}, {0.5f, 0.5f, 0.5f}, 5u),  // below frustum: culled
        makeInstance({0.0f, 0.0f, 4.98f}, {0.01f, 0.01f, 0.01f}, 6u), // strictly in front of near=0.1: culled
        makeInstance({0.0f, 0.0f, 4.90f}, {0.2f, 0.2f, 0.2f}, 7u),    // straddling near plane: visible
    };

    const CullConstants constants = makeCullConstants(vp, 64, 64, 1, static_cast<std::uint32_t>(instances.size()),
                                                      0xffffffffu, 0u, {}, false);
    const CpuCullResult res = cullInstancesPhase1Cpu(instances, constants, {});
    ASSERT_EQ(res.visibility.size(), instances.size());
    EXPECT_EQ(res.visibility[0], kCullVisibilityPhase1);
    EXPECT_EQ(res.visibility[1], kCullVisibilityCulled);
    EXPECT_EQ(res.visibility[2], kCullVisibilityCulled);
    EXPECT_EQ(res.visibility[3], kCullVisibilityCulled);
    EXPECT_EQ(res.visibility[4], kCullVisibilityCulled);
    EXPECT_EQ(res.visibility[5], kCullVisibilityCulled);
    EXPECT_EQ(res.visibility[6], kCullVisibilityCulled);
    EXPECT_EQ(res.visibility[7], kCullVisibilityPhase1);

    ASSERT_EQ(res.commands.size(), 2u);
    EXPECT_EQ(res.commands[0].firstInstance, 0u);
    EXPECT_EQ(res.commands[1].firstInstance, 7u);

    // Verify finite reversed-Z projection far-plane culling (6th plane).
    const glm::mat4 finiteProj = finitePerspectiveReverseZ(glm::radians(60.0f), 1.0f, 0.1f, 20.0f);
    const auto finitePlanes = extractReversedZFrustumPlanes(finiteProj * view.view);
    EXPECT_NEAR(glm::length(glm::vec3(finitePlanes[5])), 1.0f, 1e-5f);
    EXPECT_TRUE(intersectsReversedZFrustum(finitePlanes, {0.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}));
    EXPECT_FALSE(intersectsReversedZFrustum(finitePlanes, {0.0f, 0.0f, -25.0f}, {0.5f, 0.5f, 0.5f}));
}

TEST(GpuCulling, Phase1OccludesCubeBehindLargeWallAndEmitsIndirectDraws) {
    constexpr std::uint32_t kWidth = 256;
    constexpr std::uint32_t kHeight = 256;

    FlyCameraController cam;
    cam.fovDeg = 60.0f;
    cam.nearPlane = 0.1f;
    cam.setPosition({0.0f, 0.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const CameraView view = cam.view(1.0f);
    const glm::mat4 vp = view.projection * view.view;

    // Large opaque occluder wall centered at (0, 0, 0) covering [-2, +2] in X and Y.
    std::vector<float> depth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(depth, kWidth, kHeight, vp, {-2.0f, -2.0f, -0.1f}, {2.0f, 2.0f, 0.1f});
    const auto hizMips = buildHiZMipsCpu(depth, kWidth, kHeight);
    ASSERT_FALSE(hizMips.empty());

    const std::vector<CullInstance> instances = {
        // 0: Small cube strictly behind the wall -> occluded in Phase 1 (visibility = 2)
        makeInstance({0.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}, 10u, 36u, 0u, 0),
        // 1: Cube in front of the wall -> visible in Phase 1 (visibility = 1)
        makeInstance({0.0f, 0.0f, 2.0f}, {0.4f, 0.4f, 0.4f}, 20u, 36u, 36u, 24),
        // 2: Cube well to the right of the wall (still in frustum: at z=-3 distance=8, edge at x=3.2) -> visible in Phase 1
        makeInstance({3.8f, 0.0f, -3.0f}, {0.4f, 0.4f, 0.4f}, 30u, 72u, 108u, -8),
        // 3: Cube partially peeking past the wall edge (wall edge projects to x=4.0 at z=-5; x in [3.5, 4.5]) -> conservatively visible
        makeInstance({4.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}, 40u, 36u, 0u, 0),
        // 4: Cube behind the camera -> frustum culled (visibility = 0)
        makeInstance({0.0f, 0.0f, 12.0f}, {0.5f, 0.5f, 0.5f}, 50u, 36u, 0u, 0),
        // 5: Cube behind the wall but marked skip-occlusion -> visible in Phase 1
        makeInstance({0.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}, 60u, 36u, 0u, 0, kCullInstanceSkipOcclusion),
    };

    const CullConstants constants = makeCullConstants(
        vp, kWidth, kHeight, static_cast<std::uint32_t>(hizMips.size()),
        static_cast<std::uint32_t>(instances.size()));
    const CpuCullResult phase1 = cullInstancesPhase1Cpu(instances, constants, hizMips);

    ASSERT_EQ(phase1.visibility.size(), instances.size());
    EXPECT_EQ(phase1.visibility[0], kCullVisibilityOccluded);
    EXPECT_EQ(phase1.visibility[1], kCullVisibilityPhase1);
    EXPECT_EQ(phase1.visibility[2], kCullVisibilityPhase1);
    EXPECT_EQ(phase1.visibility[3], kCullVisibilityPhase1);
    EXPECT_EQ(phase1.visibility[4], kCullVisibilityCulled);
    EXPECT_EQ(phase1.visibility[5], kCullVisibilityPhase1);

    ASSERT_EQ(phase1.commands.size(), 4u);
    EXPECT_EQ(phase1.commands[0].firstInstance, 20u);
    EXPECT_EQ(phase1.commands[0].indexCount, 36u);
    EXPECT_EQ(phase1.commands[0].instanceCount, 1u);
    EXPECT_EQ(phase1.commands[0].firstIndex, 36u);
    EXPECT_EQ(phase1.commands[0].vertexOffset, 24);
    EXPECT_EQ(phase1.commands[1].firstInstance, 30u);
    EXPECT_EQ(phase1.commands[1].vertexOffset, -8);
    EXPECT_EQ(phase1.commands[2].firstInstance, 40u);
    EXPECT_EQ(phase1.commands[3].firstInstance, 60u);
}

TEST(GpuCulling, Phase2DisocclusionWhenOccluderWallMovesOrDepthClears) {
    constexpr std::uint32_t kWidth = 197;   // intentionally odd dimensions to exercise conservative Hi-Z
    constexpr std::uint32_t kHeight = 135;

    FlyCameraController cam;
    cam.fovDeg = 60.0f;
    cam.nearPlane = 0.1f;
    cam.setPosition({0.0f, 0.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const CameraView view = cam.view(static_cast<float>(kWidth) / static_cast<float>(kHeight));
    const glm::mat4 vp = view.projection * view.view;

    const std::vector<CullInstance> instances = {
        makeInstance({0.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}, 100u, 36u, 0u, 0),   // behind wall initially
        makeInstance({0.0f, 0.0f, 2.0f}, {0.3f, 0.3f, 0.3f}, 101u, 36u, 36u, 8),   // always visible in front
        makeInstance({0.0f, 0.0f, 15.0f}, {0.5f, 0.5f, 0.5f}, 102u, 36u, 0u, 0),   // behind camera
    };

    // Previous-frame depth has a large wall at z = 0 blocking instance 0.
    std::vector<float> prevDepth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(prevDepth, kWidth, kHeight, vp, {-2.5f, -2.5f, -0.1f}, {2.5f, 2.5f, 0.1f});
    const auto prevHiZ = buildHiZMipsCpu(prevDepth, kWidth, kHeight);

    const CullConstants prevConstants = makeCullConstants(
        vp, kWidth, kHeight, static_cast<std::uint32_t>(prevHiZ.size()),
        static_cast<std::uint32_t>(instances.size()));
    CpuCullResult phase1 = cullInstancesPhase1Cpu(instances, prevConstants, prevHiZ);
    EXPECT_EQ(phase1.visibility[0], kCullVisibilityOccluded);
    EXPECT_EQ(phase1.visibility[1], kCullVisibilityPhase1);
    EXPECT_EQ(phase1.visibility[2], kCullVisibilityCulled);
    ASSERT_EQ(phase1.commands.size(), 1u);
    EXPECT_EQ(phase1.commands[0].firstInstance, 101u);

    // If the wall is STILL blocking in current-frame Hi-Z, Phase 2 emits nothing.
    std::vector<std::uint32_t> visStillBlocked = phase1.visibility;
    const auto blockedPhase2 = cullInstancesPhase2Cpu(instances, prevConstants, prevHiZ, visStillBlocked);
    EXPECT_TRUE(blockedPhase2.empty());
    EXPECT_EQ(visStillBlocked[0], kCullVisibilityOccluded);

    // Now the occluder wall moves far aside (out of center view) and current-frame depth clears
    // around the center: Phase 2 must disocclude instance 0 without re-emitting instance 1.
    std::vector<float> currDepth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(currDepth, kWidth, kHeight, vp, {4.0f, -2.5f, -0.1f}, {9.0f, 2.5f, 0.1f});
    const auto currHiZ = buildHiZMipsCpu(currDepth, kWidth, kHeight);
    const CullConstants currConstants = makeCullConstants(
        vp, kWidth, kHeight, static_cast<std::uint32_t>(currHiZ.size()),
        static_cast<std::uint32_t>(instances.size()));

    const auto phase2 = cullInstancesPhase2Cpu(instances, currConstants, currHiZ, phase1.visibility);
    ASSERT_EQ(phase2.size(), 1u);
    EXPECT_EQ(phase2[0].firstInstance, 100u);
    EXPECT_EQ(phase2[0].indexCount, 36u);
    EXPECT_EQ(phase2[0].instanceCount, 1u);
    EXPECT_EQ(phase1.visibility[0], kCullVisibilityPhase2);
    EXPECT_EQ(phase1.visibility[1], kCullVisibilityPhase1);
    EXPECT_EQ(phase1.visibility[2], kCullVisibilityCulled);
}

TEST(GpuCulling, NearPlaneIntersectingAabbIsConservativelyVisible) {
    FlyCameraController cam;
    cam.fovDeg = 60.0f;
    cam.nearPlane = 0.5f;
    cam.setPosition({0.0f, 0.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const CameraView view = cam.view(1.0f);
    const glm::mat4 vp = view.projection * view.view;

    // Fill depth with maximum reversed-Z depth (1.0): any box crossing the near plane must stay visible.
    std::vector<float> depth(64u * 64u, 1.0f);
    const auto hizMips = buildHiZMipsCpu(depth, 64, 64);

    const std::vector<CullInstance> instances = {
        makeInstance({0.0f, 0.0f, 4.5f}, {1.0f, 1.0f, 1.0f}, 7u),  // crosses near plane & camera origin
    };
    const CullConstants constants = makeCullConstants(vp, 64, 64, static_cast<std::uint32_t>(hizMips.size()), 1u);
    const CpuCullResult res = cullInstancesPhase1Cpu(instances, constants, hizMips);
    EXPECT_EQ(res.visibility[0], kCullVisibilityPhase1);
    ASSERT_EQ(res.commands.size(), 1u);
    EXPECT_EQ(res.commands[0].firstInstance, 7u);
}

TEST(GpuCulling, GpuOcclusionCullerVulkanIntegration) {
    std::filesystem::path shaderDir;
    if (const char* overrideDir = std::getenv("LUMORA_HIZ_TEST_SHADER_DIR"); overrideDir != nullptr && *overrideDir != '\0') {
        shaderDir = overrideDir;
    } else {
        shaderDir = paths::shaderDir();
    }
    if (!std::filesystem::exists(shaderDir / "hiz.build.comp.spv") ||
        !std::filesystem::exists(shaderDir / "gpu.cull.phase1.comp.spv") ||
        !std::filesystem::exists(shaderDir / "gpu.cull.phase2.comp.spv")) {
        GTEST_SKIP() << "compiled culling shaders not present in " << shaderDir.string();
    }

    RenderContextConfig ctxCfg;
    ctxCfg.requirePresentation = false;
    ctxCfg.shaderDirectory = shaderDir;
    ctxCfg.enableValidation = std::getenv("LUMORA_TEST_VALIDATION") != nullptr;

    std::unique_ptr<RenderContext> ctx;
    try {
        ctx = std::make_unique<RenderContext>(ctxCfg);
    } catch (const rhi::UnsupportedHardwareError& e) {
        GTEST_SKIP() << "no suitable GPU: " << e.what();
    }

    constexpr std::uint32_t kWidth = 135;
    constexpr std::uint32_t kHeight = 97;

    FlyCameraController cam;
    cam.fovDeg = 60.0f;
    cam.nearPlane = 0.1f;
    cam.setPosition({0.0f, 0.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const CameraView view = cam.view(static_cast<float>(kWidth) / static_cast<float>(kHeight));
    const glm::mat4 vp = view.projection * view.view;

    const std::vector<CullInstance> instances = {
        makeInstance({0.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}, 11u, 36u, 0u, 0),
        makeInstance({0.0f, 0.0f, 2.0f}, {0.4f, 0.4f, 0.4f}, 22u, 48u, 12u, 4),
        makeInstance({0.0f, 0.0f, 15.0f}, {0.5f, 0.5f, 0.5f}, 33u, 36u, 0u, 0),
    };

    std::vector<float> prevDepth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(prevDepth, kWidth, kHeight, vp, {-2.5f, -2.5f, -0.1f}, {2.5f, 2.5f, 0.1f});

    std::vector<float> currDepth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(currDepth, kWidth, kHeight, vp, {4.5f, -2.5f, -0.1f}, {8.5f, 2.5f, 0.1f});

    // CPU reference ground truth
    const auto cpuPrevHiZ = buildHiZMipsCpu(prevDepth, kWidth, kHeight);
    const auto cpuCurrHiZ = buildHiZMipsCpu(currDepth, kWidth, kHeight);
    const CullConstants cpuPrevConst = makeCullConstants(
        vp, kWidth, kHeight, static_cast<std::uint32_t>(cpuPrevHiZ.size()),
        static_cast<std::uint32_t>(instances.size()));
    CpuCullResult cpuPhase1 = cullInstancesPhase1Cpu(instances, cpuPrevConst, cpuPrevHiZ);
    const CullConstants cpuCurrConst = makeCullConstants(
        vp, kWidth, kHeight, static_cast<std::uint32_t>(cpuCurrHiZ.size()),
        static_cast<std::uint32_t>(instances.size()));
    std::vector<std::uint32_t> cpuVisAfterPhase2 = cpuPhase1.visibility;
    const auto cpuPhase2 = cullInstancesPhase2Cpu(instances, cpuCurrConst, cpuCurrHiZ, cpuVisAfterPhase2);

    nvrhi::IDevice* gpu = ctx->gpu();
    GpuOcclusionCuller culler(*ctx);

    // Create minimal FrameResources binding set required by common.hlsli Set 0.
    FrameResources frame;
    const FrameConstants fc{};
    const GpuObject dummyObj{};
    const glm::mat4 dummyJoint{1.0f};
    const GpuLight dummyLight{};
    const std::uint32_t dummyGrid = 0u;
    frame.constants.write(gpu, std::as_bytes(std::span(&fc, 1)), DynamicBuffer::Kind::Constant, sizeof(FrameConstants), "FC");
    frame.objects.write(gpu, std::as_bytes(std::span(&dummyObj, 1)), DynamicBuffer::Kind::Structured, sizeof(GpuObject), "Obj");
    frame.joints.write(gpu, std::as_bytes(std::span(&dummyJoint, 1)), DynamicBuffer::Kind::Structured, sizeof(glm::mat4), "Jnt");
    frame.pointLights.write(gpu, std::as_bytes(std::span(&dummyLight, 1)), DynamicBuffer::Kind::Structured, sizeof(GpuLight), "Lgt");
    frame.lightGrid.write(gpu, std::as_bytes(std::span(&dummyGrid, 1)), DynamicBuffer::Kind::Structured, sizeof(std::uint32_t), "Grd");
    nvrhi::IBindingSet* frameBindings = frame.bindingSet(gpu, ctx->frameLayout(), ctx->samplers());

    nvrhi::TextureDesc depthDesc;
    depthDesc.width = kWidth;
    depthDesc.height = kHeight;
    depthDesc.format = nvrhi::Format::R32_FLOAT;
    depthDesc.isShaderResource = true;
    depthDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    depthDesc.keepInitialState = true;
    depthDesc.debugName = "TestDepthInput";
    nvrhi::TextureHandle depthTex = gpu->createTexture(depthDesc);
    ASSERT_TRUE(depthTex);
    const std::uint32_t depthBindless = ctx->bindless().registerTexture(depthTex);

    // Phase 1 on GPU with previous-frame depth
    ctx->upload([&](nvrhi::ICommandList* cmd) {
        cmd->writeTexture(depthTex, 0, 0, prevDepth.data(), kWidth * sizeof(float));
        culler.uploadInstances(cmd, instances);
        culler.buildHiZ(cmd, frameBindings, depthTex, depthBindless, kWidth, kHeight);
        culler.dispatchPhase1(cmd, frameBindings, vp, static_cast<std::uint32_t>(instances.size()), true);
    });
    gpu->waitForIdle();

    const auto gpuCount1 = readBufferVec<std::uint32_t>(*ctx, culler.phase1CountBuffer(), 1u);
    const auto gpuVis1 = readBufferVec<std::uint32_t>(*ctx, culler.visibilityBuffer(), instances.size());
    ASSERT_EQ(gpuCount1[0], cpuPhase1.commands.size());
    EXPECT_EQ(gpuVis1, cpuPhase1.visibility);
    if (gpuCount1[0] > 0u) {
        const auto gpuCmds1 = readBufferVec<DrawIndexedIndirectCommand>(
            *ctx, culler.phase1IndirectBuffer(), gpuCount1[0]);
        for (std::size_t i = 0; i < gpuCmds1.size(); ++i) {
            EXPECT_EQ(gpuCmds1[i].indexCount, cpuPhase1.commands[i].indexCount);
            EXPECT_EQ(gpuCmds1[i].instanceCount, cpuPhase1.commands[i].instanceCount);
            EXPECT_EQ(gpuCmds1[i].firstIndex, cpuPhase1.commands[i].firstIndex);
            EXPECT_EQ(gpuCmds1[i].vertexOffset, cpuPhase1.commands[i].vertexOffset);
            EXPECT_EQ(gpuCmds1[i].firstInstance, cpuPhase1.commands[i].firstInstance);
        }
    }

    // Phase 2 on GPU after occluder wall moves away in current-frame depth
    ctx->upload([&](nvrhi::ICommandList* cmd) {
        cmd->writeTexture(depthTex, 0, 0, currDepth.data(), kWidth * sizeof(float));
        culler.buildHiZ(cmd, frameBindings, depthTex, depthBindless, kWidth, kHeight);
        culler.dispatchPhase2(cmd, frameBindings, vp, static_cast<std::uint32_t>(instances.size()));
    });
    gpu->waitForIdle();

    const auto gpuCount2 = readBufferVec<std::uint32_t>(*ctx, culler.phase2CountBuffer(), 1u);
    const auto gpuVis2 = readBufferVec<std::uint32_t>(*ctx, culler.visibilityBuffer(), instances.size());
    ASSERT_EQ(gpuCount2[0], cpuPhase2.size());
    EXPECT_EQ(gpuVis2, cpuVisAfterPhase2);
    if (gpuCount2[0] > 0u) {
        const auto gpuCmds2 = readBufferVec<DrawIndexedIndirectCommand>(
            *ctx, culler.phase2IndirectBuffer(), gpuCount2[0]);
        for (std::size_t i = 0; i < gpuCmds2.size(); ++i) {
            EXPECT_EQ(gpuCmds2[i].indexCount, cpuPhase2[i].indexCount);
            EXPECT_EQ(gpuCmds2[i].instanceCount, cpuPhase2[i].instanceCount);
            EXPECT_EQ(gpuCmds2[i].firstIndex, cpuPhase2[i].firstIndex);
            EXPECT_EQ(gpuCmds2[i].vertexOffset, cpuPhase2[i].vertexOffset);
            EXPECT_EQ(gpuCmds2[i].firstInstance, cpuPhase2[i].firstInstance);
        }
    }

    ctx->bindless().releaseTexture(depthBindless);
}

TEST(GpuCulling, SlotIndirectBuffersMatchTwoPhaseVisibility) {
    std::filesystem::path shaderDir;
    if (const char* overrideDir = std::getenv("LUMORA_HIZ_TEST_SHADER_DIR"); overrideDir != nullptr && *overrideDir != '\0') {
        shaderDir = overrideDir;
    } else {
        shaderDir = paths::shaderDir();
    }
    if (!std::filesystem::exists(shaderDir / "hiz.build.comp.spv") ||
        !std::filesystem::exists(shaderDir / "gpu.cull.phase1.comp.spv") ||
        !std::filesystem::exists(shaderDir / "gpu.cull.phase2.comp.spv")) {
        GTEST_SKIP() << "compiled culling shaders not present in " << shaderDir.string();
    }

    RenderContextConfig ctxCfg;
    ctxCfg.requirePresentation = false;
    ctxCfg.shaderDirectory = shaderDir;
    ctxCfg.enableValidation = std::getenv("LUMORA_TEST_VALIDATION") != nullptr;

    std::unique_ptr<RenderContext> ctx;
    try {
        ctx = std::make_unique<RenderContext>(ctxCfg);
    } catch (const rhi::UnsupportedHardwareError& e) {
        GTEST_SKIP() << "no suitable GPU: " << e.what();
    }

    constexpr std::uint32_t kWidth = 135;
    constexpr std::uint32_t kHeight = 97;

    FlyCameraController cam;
    cam.fovDeg = 60.0f;
    cam.nearPlane = 0.1f;
    cam.setPosition({0.0f, 0.0f, 5.0f});
    cam.lookAt({0.0f, 0.0f, 0.0f});
    const CameraView view = cam.view(static_cast<float>(kWidth) / static_cast<float>(kHeight));
    const glm::mat4 vp = view.projection * view.view;

    std::vector<CullInstance> instances = {
        makeInstance({0.0f, 0.0f, -5.0f}, {0.5f, 0.5f, 0.5f}, 11u, 36u, 0u, 0, 0u),
        makeInstance({0.0f, 0.0f, 2.0f}, {0.4f, 0.4f, 0.4f}, 22u, 48u, 12u, 4, 0u),
        makeInstance({0.0f, 0.0f, 15.0f}, {0.5f, 0.5f, 0.5f}, 33u, 36u, 0u, 0, 0u),
        makeInstance({0.0f, 0.0f, -6.0f}, {0.5f, 0.5f, 0.5f}, 44u, 36u, 0u, 0, kCullInstanceSkipOcclusion),
    };
    instances[0].reserved = 1u;
    instances[1].reserved = 1u;
    instances[2].reserved = 2u;
    instances[3].reserved = 3u;

    std::vector<float> prevDepth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(prevDepth, kWidth, kHeight, vp, {-2.5f, -2.5f, -0.1f}, {2.5f, 2.5f, 0.1f});

    std::vector<float> currDepth(static_cast<std::size_t>(kWidth) * kHeight, 0.0f);
    rasterizeWallToDepthBuffer(currDepth, kWidth, kHeight, vp, {4.5f, -2.5f, -0.1f}, {8.5f, 2.5f, 0.1f});

    nvrhi::IDevice* gpu = ctx->gpu();
    GpuOcclusionCuller culler(*ctx);

    FrameResources frame;
    const FrameConstants fc{};
    const GpuObject dummyObj{};
    const glm::mat4 dummyJoint{1.0f};
    const GpuLight dummyLight{};
    const std::uint32_t dummyGrid = 0u;
    frame.constants.write(gpu, std::as_bytes(std::span(&fc, 1)), DynamicBuffer::Kind::Constant, sizeof(FrameConstants), "FC");
    frame.objects.write(gpu, std::as_bytes(std::span(&dummyObj, 1)), DynamicBuffer::Kind::Structured, sizeof(GpuObject), "Obj");
    frame.joints.write(gpu, std::as_bytes(std::span(&dummyJoint, 1)), DynamicBuffer::Kind::Structured, sizeof(glm::mat4), "Jnt");
    frame.pointLights.write(gpu, std::as_bytes(std::span(&dummyLight, 1)), DynamicBuffer::Kind::Structured, sizeof(GpuLight), "Lgt");
    frame.lightGrid.write(gpu, std::as_bytes(std::span(&dummyGrid, 1)), DynamicBuffer::Kind::Structured, sizeof(std::uint32_t), "Grd");
    nvrhi::IBindingSet* frameBindings = frame.bindingSet(gpu, ctx->frameLayout(), ctx->samplers());

    nvrhi::TextureDesc depthDesc;
    depthDesc.width = kWidth;
    depthDesc.height = kHeight;
    depthDesc.format = nvrhi::Format::R32_FLOAT;
    depthDesc.isShaderResource = true;
    depthDesc.initialState = nvrhi::ResourceStates::ShaderResource;
    depthDesc.keepInitialState = true;
    depthDesc.debugName = "TestSlotDepth";
    nvrhi::TextureHandle depthTex = gpu->createTexture(depthDesc);
    ASSERT_TRUE(depthTex);
    const std::uint32_t depthBindless = ctx->bindless().registerTexture(depthTex);

    ctx->upload([&](nvrhi::ICommandList* cmd) {
        cmd->writeTexture(depthTex, 0, 0, prevDepth.data(), kWidth * sizeof(float));
        culler.uploadInstances(cmd, instances);
        culler.buildHiZ(cmd, frameBindings, depthTex, depthBindless, kWidth, kHeight);
        culler.dispatchPhase1(cmd, frameBindings, vp, static_cast<std::uint32_t>(instances.size()), true);
    });
    gpu->waitForIdle();

    const auto vis1 = readBufferVec<std::uint32_t>(*ctx, culler.visibilityBuffer(), instances.size());
    const auto p1Slots = readBufferVec<DrawIndexedIndirectCommand>(*ctx, culler.phase1SlotIndirectBuffer(), instances.size());
    const auto fSlots1 = readBufferVec<DrawIndexedIndirectCommand>(*ctx, culler.finalSlotIndirectBuffer(), instances.size());
    EXPECT_EQ(vis1[0], kCullVisibilityOccluded);
    EXPECT_EQ(vis1[1], kCullVisibilityPhase1);
    EXPECT_EQ(vis1[2], kCullVisibilityCulled);
    EXPECT_EQ(vis1[3], kCullVisibilityPhase1);
    EXPECT_EQ(p1Slots[0].instanceCount, 0u);
    EXPECT_EQ(p1Slots[1].instanceCount, 1u);
    EXPECT_EQ(p1Slots[2].instanceCount, 0u);
    EXPECT_EQ(p1Slots[3].instanceCount, 3u);
    for (std::size_t i = 0; i < instances.size(); ++i) {
        EXPECT_EQ(fSlots1[i].instanceCount, p1Slots[i].instanceCount);
        EXPECT_EQ(p1Slots[i].firstInstance, 0u);
    }

    ctx->upload([&](nvrhi::ICommandList* cmd) {
        cmd->writeTexture(depthTex, 0, 0, currDepth.data(), kWidth * sizeof(float));
        culler.buildHiZ(cmd, frameBindings, depthTex, depthBindless, kWidth, kHeight);
        culler.dispatchPhase2(cmd, frameBindings, vp, static_cast<std::uint32_t>(instances.size()));
    });
    gpu->waitForIdle();

    const auto vis2 = readBufferVec<std::uint32_t>(*ctx, culler.visibilityBuffer(), instances.size());
    const auto p2Slots = readBufferVec<DrawIndexedIndirectCommand>(*ctx, culler.phase2SlotIndirectBuffer(), instances.size());
    const auto fSlots2 = readBufferVec<DrawIndexedIndirectCommand>(*ctx, culler.finalSlotIndirectBuffer(), instances.size());
    EXPECT_EQ(vis2[0], kCullVisibilityPhase2);
    EXPECT_EQ(vis2[1], kCullVisibilityPhase1);
    EXPECT_EQ(vis2[2], kCullVisibilityCulled);
    EXPECT_EQ(vis2[3], kCullVisibilityPhase1);
    EXPECT_EQ(p2Slots[0].instanceCount, 1u);
    EXPECT_EQ(p2Slots[1].instanceCount, 0u);
    EXPECT_EQ(p2Slots[2].instanceCount, 0u);
    EXPECT_EQ(p2Slots[3].instanceCount, 0u);
    EXPECT_EQ(fSlots2[0].instanceCount, 1u);
    EXPECT_EQ(fSlots2[1].instanceCount, 1u);
    EXPECT_EQ(fSlots2[2].instanceCount, 0u);
    EXPECT_EQ(fSlots2[3].instanceCount, 3u);

    ctx->bindless().releaseTexture(depthBindless);
}
