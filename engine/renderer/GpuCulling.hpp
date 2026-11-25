#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <nvrhi/nvrhi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lumora::render {

class RenderContext;

inline constexpr std::uint32_t kCullVisibilityCulled = 0u;
inline constexpr std::uint32_t kCullVisibilityPhase1 = 1u;
inline constexpr std::uint32_t kCullVisibilityOccluded = 2u;
inline constexpr std::uint32_t kCullVisibilityPhase2 = 3u;

inline constexpr std::uint32_t kCullFlagFrustum = 1u << 0;
inline constexpr std::uint32_t kCullFlagOcclusion = 1u << 1;
inline constexpr std::uint32_t kCullFlagHiZValid = 1u << 2;

inline constexpr std::uint32_t kCullInstanceSkipOcclusion = 1u << 0;
inline constexpr std::uint32_t kCullInstanceSkipFrustum = 1u << 1;

/// Per-instance culling input (matches CullInstance in shaders/gpu_cull.hlsl).
struct alignas(16) CullInstance {
    glm::vec3 aabbCenter{0.0f};
    std::uint32_t indexCount = 0;
    glm::vec3 aabbExtents{0.0f};
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t objectIndex = 0;
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;  // optional instance count (> 0) for per-slot indirect draws
};
static_assert(sizeof(CullInstance) == 48);
static_assert(alignof(CullInstance) == 16);
static_assert(offsetof(CullInstance, aabbCenter) == 0);
static_assert(offsetof(CullInstance, indexCount) == 12);
static_assert(offsetof(CullInstance, aabbExtents) == 16);
static_assert(offsetof(CullInstance, firstIndex) == 28);
static_assert(offsetof(CullInstance, vertexOffset) == 32);
static_assert(offsetof(CullInstance, objectIndex) == 36);
static_assert(offsetof(CullInstance, flags) == 40);
static_assert(offsetof(CullInstance, reserved) == 44);

/// Indexed indirect draw command (matches VkDrawIndexedIndirectCommand, D3D12, and shaders/gpu_cull.hlsl).
struct DrawIndexedIndirectCommand {
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t firstInstance = 0;
};
static_assert(sizeof(DrawIndexedIndirectCommand) == 20);
static_assert(alignof(DrawIndexedIndirectCommand) == 4);
static_assert(sizeof(DrawIndexedIndirectCommand) == sizeof(nvrhi::DrawIndexedIndirectArguments));
static_assert(offsetof(DrawIndexedIndirectCommand, indexCount) == 0);
static_assert(offsetof(DrawIndexedIndirectCommand, instanceCount) == 4);
static_assert(offsetof(DrawIndexedIndirectCommand, firstIndex) == 8);
static_assert(offsetof(DrawIndexedIndirectCommand, vertexOffset) == 12);
static_assert(offsetof(DrawIndexedIndirectCommand, firstInstance) == 16);

/// Constant buffer layout for CSPhase1 and CSPhase2 (matches CullConstants in shaders/gpu_cull.hlsl).
struct alignas(16) CullConstants {
    glm::mat4 viewProjection{1.0f};
    glm::vec4 frustumPlanes[6]{};
    glm::vec2 hizSize{0.0f};
    glm::vec2 invHizSize{0.0f};
    std::uint32_t instanceCount = 0;
    std::uint32_t hizMipCount = 0;
    std::uint32_t hizTextureIndex = 0xffffffffu;
    std::uint32_t hizSamplerIndex = 0;
    float nearPlane = 0.05f;
    float depthBias = 0.0f;
    std::uint32_t flags = kCullFlagFrustum | kCullFlagOcclusion | kCullFlagHiZValid;
    std::uint32_t reserved = 0;
};
static_assert(sizeof(CullConstants) == 208);
static_assert(alignof(CullConstants) == 16);
static_assert(offsetof(CullConstants, viewProjection) == 0);
static_assert(offsetof(CullConstants, frustumPlanes) == 64);
static_assert(offsetof(CullConstants, hizSize) == 160);
static_assert(offsetof(CullConstants, invHizSize) == 168);
static_assert(offsetof(CullConstants, instanceCount) == 176);
static_assert(offsetof(CullConstants, hizMipCount) == 180);
static_assert(offsetof(CullConstants, hizTextureIndex) == 184);
static_assert(offsetof(CullConstants, hizSamplerIndex) == 188);
static_assert(offsetof(CullConstants, nearPlane) == 192);
static_assert(offsetof(CullConstants, depthBias) == 196);
static_assert(offsetof(CullConstants, flags) == 200);
static_assert(offsetof(CullConstants, reserved) == 204);

/// Configuration options for two-phase GPU/CPU occlusion culling.
struct CullConfig {
    bool frustumCulling = true;
    bool occlusionCulling = true;
    float nearPlane = 0.05f;
    float depthBias = 0.0f;
};

/// Single mip level in the CPU reference Hi-Z depth pyramid (row-major).
struct HiZMipLevel {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> depths;
};

/// Output of CPU Phase-1 culling, matching GPU CSPhase1 buffers.
struct CpuCullResult {
    std::vector<DrawIndexedIndirectCommand> commands;
    std::vector<std::uint32_t> visibility;
};

/// Extracts 6 normalized inward-pointing frustum planes (Left, Right, Top, Bottom, Near, Far)
/// from a Vulkan Y-down reversed-Z (near = 1, far = 0) view-projection matrix.
[[nodiscard]] std::array<glm::vec4, 6> extractReversedZFrustumPlanes(const glm::mat4& vp) noexcept;

/// Tests whether a world-space AABB intersects or lies inside all 6 frustum planes.
[[nodiscard]] bool intersectsReversedZFrustum(std::span<const glm::vec4, 6> planes,
                                              const glm::vec3& center,
                                              const glm::vec3& extents) noexcept;

/// Returns the mip count of a full Hi-Z pyramid from (width, height) down to 1x1.
[[nodiscard]] std::uint32_t computeHiZMipCount(std::uint32_t width, std::uint32_t height) noexcept;

/// Builds a CullConstants block with extracted frustum planes and Hi-Z pyramid metadata.
[[nodiscard]] CullConstants makeCullConstants(const glm::mat4& viewProjection,
                                              std::uint32_t hizWidth,
                                              std::uint32_t hizHeight,
                                              std::uint32_t hizMipCount,
                                              std::uint32_t instanceCount,
                                              std::uint32_t hizTextureIndex = 0xffffffffu,
                                              std::uint32_t hizSamplerIndex = 0,
                                              const CullConfig& config = {},
                                              bool hizValid = true,
                                              const glm::mat4* hizViewProjection = nullptr) noexcept;

/// CPU reference reversed-Z Hi-Z mip builder matching shaders/hiz_build.hlsl bit-for-bit.
/// Performs conservative min-reduction with extra boundary taps when source dimensions are odd.
[[nodiscard]] std::vector<HiZMipLevel> buildHiZMipsCpu(std::span<const float> mip0Depths,
                                                       std::uint32_t width,
                                                       std::uint32_t height);

/// CPU reference Phase-1 culling matching shaders/gpu_cull.hlsl::CSPhase1.
[[nodiscard]] CpuCullResult cullInstancesPhase1Cpu(std::span<const CullInstance> instances,
                                                   const CullConstants& constants,
                                                   std::span<const HiZMipLevel> hizMips);

/// CPU reference Phase-2 culling matching shaders/gpu_cull.hlsl::CSPhase2.
/// Checks only instances with visibility[i] == kCullVisibilityOccluded against the current-frame
/// Hi-Z pyramid; newly visible instances emit commands and update visibility[i] to kCullVisibilityPhase2.
[[nodiscard]] std::vector<DrawIndexedIndirectCommand> cullInstancesPhase2Cpu(
    std::span<const CullInstance> instances,
    const CullConstants& constants,
    std::span<const HiZMipLevel> hizMips,
    std::span<std::uint32_t> visibility);

/// NVRHI helper managing Hi-Z pyramid generation and two-phase GPU occlusion culling resources.
class GpuOcclusionCuller {
public:
    explicit GpuOcclusionCuller(RenderContext& ctx);
    ~GpuOcclusionCuller();

    GpuOcclusionCuller(const GpuOcclusionCuller&) = delete;
    GpuOcclusionCuller& operator=(const GpuOcclusionCuller&) = delete;

    /// Ensures the R32_FLOAT Hi-Z pyramid exists at (width, height) and is registered in BindlessTable.
    void ensureHiZResources(std::uint32_t width, std::uint32_t height);

    /// Ensures GPU instance, visibility, indirect command, and draw-count buffers hold >= maxInstances.
    void ensureInstanceBuffers(std::uint32_t maxInstances);

    /// Uploads instance descriptors into the GPU structured buffer.
    void uploadInstances(nvrhi::ICommandList* cmd, std::span<const CullInstance> instances);

    /// Builds the full reversed-Z Hi-Z mip pyramid from `depthTexture` (shaders/hiz_build.hlsl).
    void buildHiZ(nvrhi::ICommandList* cmd,
                  nvrhi::IBindingSet* frameBindings,
                  nvrhi::ITexture* depthTexture,
                  std::uint32_t depthBindlessIndex,
                  std::uint32_t width,
                  std::uint32_t height);

    /// Dispatches Phase-1 culling (CSPhase1) against the stored (previous-frame) Hi-Z pyramid.
    void dispatchPhase1(nvrhi::ICommandList* cmd,
                        nvrhi::IBindingSet* frameBindings,
                        const glm::mat4& viewProjection,
                        std::uint32_t instanceCount,
                        bool hizValid = true,
                        const CullConfig& config = {},
                        const glm::mat4* hizViewProjection = nullptr);

    /// Dispatches Phase-2 culling (CSPhase2) for visibility==2 candidates against the current-frame Hi-Z pyramid.
    void dispatchPhase2(nvrhi::ICommandList* cmd,
                        nvrhi::IBindingSet* frameBindings,
                        const glm::mat4& viewProjection,
                        std::uint32_t instanceCount,
                        const CullConfig& config = {});

    [[nodiscard]] nvrhi::ITexture* hizTexture() const noexcept { return hizTexture_; }
    [[nodiscard]] std::uint32_t hizBindlessIndex() const noexcept { return hizBindlessIndex_; }
    [[nodiscard]] std::uint32_t hizWidth() const noexcept { return hizWidth_; }
    [[nodiscard]] std::uint32_t hizHeight() const noexcept { return hizHeight_; }
    [[nodiscard]] std::uint32_t hizMipCount() const noexcept { return hizMipCount_; }

    [[nodiscard]] nvrhi::IBuffer* instanceBuffer() const noexcept { return instanceBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* visibilityBuffer() const noexcept { return visibilityBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* phase1IndirectBuffer() const noexcept { return phase1IndirectBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* phase1CountBuffer() const noexcept { return phase1CountBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* phase2IndirectBuffer() const noexcept { return phase2IndirectBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* phase2CountBuffer() const noexcept { return phase2CountBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* phase1SlotIndirectBuffer() const noexcept { return phase1SlotIndirectBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* phase2SlotIndirectBuffer() const noexcept { return phase2SlotIndirectBuffer_; }
    [[nodiscard]] nvrhi::IBuffer* finalSlotIndirectBuffer() const noexcept { return finalSlotIndirectBuffer_; }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return instanceCapacity_; }
    [[nodiscard]] nvrhi::IBindingLayout* cullLayout() const noexcept { return cullLayout_; }

private:
    void releaseHiZResources();
    void dispatchCullPass(nvrhi::ICommandList* cmd,
                          nvrhi::IComputePipeline* pipeline,
                          nvrhi::IBuffer* constantsBuffer,
                          nvrhi::IBuffer* countBuffer,
                          nvrhi::IBindingSet* cullSet,
                          nvrhi::IBindingSet* frameBindings,
                          const glm::mat4& viewProjection,
                          std::uint32_t instanceCount,
                          bool hizValid,
                          const CullConfig& config,
                          const glm::mat4* hizViewProjection = nullptr);

    RenderContext& ctx_;
    nvrhi::BindingLayoutHandle cullLayout_;
    nvrhi::ComputePipelineHandle hizBuildPipeline_;
    nvrhi::ComputePipelineHandle cullPhase1Pipeline_;
    nvrhi::ComputePipelineHandle cullPhase2Pipeline_;

    nvrhi::TextureHandle hizTexture_;
    std::uint32_t hizBindlessIndex_ = 0xffffffffu;
    std::uint32_t hizWidth_ = 0;
    std::uint32_t hizHeight_ = 0;
    std::uint32_t hizMipCount_ = 0;
    std::vector<nvrhi::BindingSetHandle> hizMipStorageSets_;

    std::uint32_t instanceCapacity_ = 0;
    nvrhi::BufferHandle instanceBuffer_;
    nvrhi::BufferHandle visibilityBuffer_;
    nvrhi::BufferHandle phase1IndirectBuffer_;
    nvrhi::BufferHandle phase1CountBuffer_;
    nvrhi::BufferHandle phase2IndirectBuffer_;
    nvrhi::BufferHandle phase2CountBuffer_;
    nvrhi::BufferHandle phase1SlotIndirectBuffer_;
    nvrhi::BufferHandle phase2SlotIndirectBuffer_;
    nvrhi::BufferHandle finalSlotIndirectBuffer_;
    nvrhi::BufferHandle phase1ConstantsBuffer_;
    nvrhi::BufferHandle phase2ConstantsBuffer_;
    nvrhi::BindingSetHandle phase1CullSet_;
    nvrhi::BindingSetHandle phase2CullSet_;
};

} // namespace lumora::render
