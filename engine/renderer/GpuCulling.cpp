#include "renderer/GpuCulling.hpp"

#include "renderer/GpuTypes.hpp"
#include "renderer/RenderContext.hpp"
#include "renderer/RenderGraph.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <bit>
#include <cmath>

namespace lumora::render {

namespace {

[[nodiscard]] DrawIndexedIndirectCommand makeIndirectCommand(const CullInstance& inst) noexcept {
    DrawIndexedIndirectCommand cmd;
    cmd.indexCount = inst.indexCount;
    cmd.instanceCount = 1u;
    cmd.firstIndex = inst.firstIndex;
    cmd.vertexOffset = inst.vertexOffset;
    cmd.firstInstance = inst.objectIndex;
    return cmd;
}

[[nodiscard]] bool isAabbVisibleInHiZCpu(const CullInstance& inst,
                                         const CullConstants& constants,
                                         std::span<const HiZMipLevel> hizMips) noexcept {
    if ((constants.flags & kCullFlagOcclusion) == 0u || (constants.flags & kCullFlagHiZValid) == 0u) {
        return true;
    }
    if ((inst.flags & kCullInstanceSkipOcclusion) != 0u) {
        return true;
    }
    if (hizMips.empty() || constants.hizMipCount == 0u || constants.hizSize.x <= 0.0f || constants.hizSize.y <= 0.0f) {
        return true;
    }

    const std::uint32_t availableMips = std::min<std::uint32_t>(
        constants.hizMipCount, static_cast<std::uint32_t>(hizMips.size()));
    if (availableMips == 0u) {
        return true;
    }

    glm::vec2 uvMin(1.0f, 1.0f);
    glm::vec2 uvMax(0.0f, 0.0f);
    float maxZ = 0.0f;

    for (std::uint32_t c = 0u; c < 8u; ++c) {
        const float sx = (c & 1u) != 0u ? 1.0f : -1.0f;
        const float sy = (c & 2u) != 0u ? 1.0f : -1.0f;
        const float sz = (c & 4u) != 0u ? 1.0f : -1.0f;
        const glm::vec3 corner = inst.aabbCenter + inst.aabbExtents * glm::vec3(sx, sy, sz);
        const glm::vec4 clip = constants.viewProjection * glm::vec4(corner, 1.0f);
        if (clip.w <= 0.0f || clip.z >= clip.w) {
            return true;
        }
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        const glm::vec2 uv = glm::vec2(ndc.x, ndc.y) * 0.5f + 0.5f;
        uvMin = glm::min(uvMin, uv);
        uvMax = glm::max(uvMax, uv);
        maxZ = std::max(maxZ, std::clamp(ndc.z, 0.0f, 1.0f));
    }

    if (uvMax.x < 0.0f || uvMin.x > 1.0f || uvMax.y < 0.0f || uvMin.y > 1.0f) {
        return true;
    }

    uvMin = glm::clamp(uvMin, glm::vec2(0.0f), glm::vec2(1.0f));
    uvMax = glm::clamp(uvMax, glm::vec2(0.0f), glm::vec2(1.0f));

    const glm::vec2 rectPx = (uvMax - uvMin) * constants.hizSize;
    const float maxDim = std::max(rectPx.x, rectPx.y);
    int mip = static_cast<int>(std::ceil(std::log2(std::max(maxDim, 1.0f))));
    mip = std::clamp(mip, 0, static_cast<int>(availableMips) - 1);

    const glm::ivec2 baseSize(static_cast<int>(constants.hizSize.x), static_cast<int>(constants.hizSize.y));
    if (mip > 0) {
        const int prevMip = mip - 1;
        const glm::ivec2 prevSize = glm::max(baseSize >> prevMip, glm::ivec2(1, 1));
        const glm::ivec2 pMin = glm::clamp(glm::ivec2(glm::floor(uvMin * glm::vec2(prevSize))),
                                           glm::ivec2(0, 0), prevSize - 1);
        const glm::ivec2 pMax = glm::clamp(glm::ivec2(glm::floor(uvMax * glm::vec2(prevSize))),
                                           glm::ivec2(0, 0), prevSize - 1);
        if (pMax.x - pMin.x <= 1 && pMax.y - pMin.y <= 1) {
            mip = prevMip;
        }
    }

    const HiZMipLevel& level = hizMips[static_cast<std::size_t>(mip)];
    if (level.width == 0u || level.height == 0u ||
        level.depths.size() < static_cast<std::size_t>(level.width) * level.height) {
        return true;
    }

    const glm::ivec2 mipSize(static_cast<int>(level.width), static_cast<int>(level.height));
    const glm::ivec2 tMin = glm::clamp(glm::ivec2(glm::floor(uvMin * glm::vec2(mipSize))),
                                       glm::ivec2(0, 0), mipSize - 1);
    const glm::ivec2 tMax = glm::clamp(glm::ivec2(glm::floor(uvMax * glm::vec2(mipSize))),
                                       glm::ivec2(0, 0), mipSize - 1);
    if (tMax.x - tMin.x > 1 || tMax.y - tMin.y > 1) {
        return true;
    }

    auto loadTexel = [&](int x, int y) noexcept -> float {
        return level.depths[static_cast<std::size_t>(y) * level.width + static_cast<std::size_t>(x)];
    };

    const float d00 = loadTexel(tMin.x, tMin.y);
    const float d10 = loadTexel(tMax.x, tMin.y);
    const float d01 = loadTexel(tMin.x, tMax.y);
    const float d11 = loadTexel(tMax.x, tMax.y);
    const float hizMinDepth = std::min({d00, d10, d01, d11});

    return !(maxZ + constants.depthBias < hizMinDepth);
}

[[nodiscard]] nvrhi::BufferHandle createGpuBuffer(nvrhi::IDevice* gpu,
                                                  std::uint64_t byteSize,
                                                  std::uint32_t structStride,
                                                  bool isConstant,
                                                  bool isIndirect,
                                                  bool canHaveUAV,
                                                  nvrhi::ResourceStates initialState,
                                                  const char* name) {
    nvrhi::BufferDesc desc;
    desc.byteSize = byteSize;
    desc.structStride = structStride;
    desc.isConstantBuffer = isConstant;
    desc.isDrawIndirectArgs = isIndirect;
    desc.canHaveUAVs = canHaveUAV;
    desc.initialState = initialState;
    desc.keepInitialState = true;
    desc.debugName = name;
    nvrhi::BufferHandle handle = gpu->createBuffer(desc);
    if (!handle) {
        throw rhi::VulkanError(std::string("could not create GPU buffer ") + name, VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    return handle;
}

[[nodiscard]] nvrhi::BindingSetHandle createCullBindingSet(nvrhi::IDevice* gpu,
                                                           nvrhi::IBindingLayout* layout,
                                                           nvrhi::IBuffer* constantsBuffer,
                                                           nvrhi::IBuffer* instanceBuffer,
                                                           nvrhi::IBuffer* indirectBuffer,
                                                           nvrhi::IBuffer* visibilityBuffer,
                                                           nvrhi::IBuffer* countBuffer,
                                                           nvrhi::IBuffer* slotIndirectBuffer,
                                                           nvrhi::IBuffer* finalSlotIndirectBuffer) {
    nvrhi::BindingSetDesc desc;
    desc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, constantsBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(1, instanceBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(2, indirectBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(3, visibilityBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(4, countBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(5, slotIndirectBuffer),
        nvrhi::BindingSetItem::StructuredBuffer_UAV(6, finalSlotIndirectBuffer),
    };
    nvrhi::BindingSetHandle set = gpu->createBindingSet(desc, layout);
    if (!set) {
        throw rhi::VulkanError("could not create GPU culling binding set", VK_ERROR_INITIALIZATION_FAILED);
    }
    return set;
}

} // namespace

std::array<glm::vec4, 6> extractReversedZFrustumPlanes(const glm::mat4& vp) noexcept {
    auto row = [&](int i) noexcept { return glm::vec4(vp[0][i], vp[1][i], vp[2][i], vp[3][i]); };
    std::array<glm::vec4, 6> planes{};
    planes[0] = row(3) + row(0);  // Left
    planes[1] = row(3) - row(0);  // Right
    planes[2] = row(3) + row(1);  // Top (Vulkan Y-down clip space)
    planes[3] = row(3) - row(1);  // Bottom
    planes[4] = row(3) - row(2);  // Near (reversed-Z: z <= w)
    planes[5] = row(2);           // Far  (reversed-Z: z >= 0)
    for (glm::vec4& p : planes) {
        const float len = glm::length(glm::vec3(p));
        if (len > 1e-12f) {
            p /= len;
        }
    }
    return planes;
}

bool intersectsReversedZFrustum(std::span<const glm::vec4, 6> planes,
                                const glm::vec3& center,
                                const glm::vec3& extents) noexcept {
    for (const glm::vec4& p : planes) {
        const glm::vec3 n(p);
        const float radius = glm::dot(glm::abs(n), extents);
        if (glm::dot(n, center) + p.w < -radius) {
            return false;
        }
    }
    return true;
}

std::uint32_t computeHiZMipCount(std::uint32_t width, std::uint32_t height) noexcept {
    const std::uint32_t maxDim = std::max(width, height);
    if (maxDim == 0u) {
        return 0u;
    }
    return static_cast<std::uint32_t>(std::bit_width(maxDim));
}

CullConstants makeCullConstants(const glm::mat4& viewProjection,
                                std::uint32_t hizWidth,
                                std::uint32_t hizHeight,
                                std::uint32_t hizMipCount,
                                std::uint32_t instanceCount,
                                std::uint32_t hizTextureIndex,
                                std::uint32_t hizSamplerIndex,
                                const CullConfig& config,
                                bool hizValid,
                                const glm::mat4* hizViewProjection) noexcept {
    CullConstants c;
    c.viewProjection = hizViewProjection != nullptr ? *hizViewProjection : viewProjection;
    const auto planes = extractReversedZFrustumPlanes(viewProjection);
    for (std::size_t i = 0; i < 6; ++i) {
        c.frustumPlanes[i] = planes[i];
    }
    c.hizSize = glm::vec2(static_cast<float>(hizWidth), static_cast<float>(hizHeight));
    c.invHizSize = glm::vec2(hizWidth > 0u ? 1.0f / static_cast<float>(hizWidth) : 0.0f,
                             hizHeight > 0u ? 1.0f / static_cast<float>(hizHeight) : 0.0f);
    c.instanceCount = instanceCount;
    c.hizMipCount = hizMipCount;
    c.hizTextureIndex = hizTextureIndex;
    c.hizSamplerIndex = hizSamplerIndex;
    c.nearPlane = config.nearPlane;
    c.depthBias = config.depthBias;
    c.flags = (config.frustumCulling ? kCullFlagFrustum : 0u) |
              (config.occlusionCulling ? kCullFlagOcclusion : 0u) |
              (hizValid ? kCullFlagHiZValid : 0u);
    c.reserved = 0u;
    return c;
}

std::vector<HiZMipLevel> buildHiZMipsCpu(std::span<const float> mip0Depths,
                                         std::uint32_t width,
                                         std::uint32_t height) {
    if (width == 0u || height == 0u ||
        mip0Depths.size() < static_cast<std::size_t>(width) * height) {
        return {};
    }

    std::vector<HiZMipLevel> mips;
    mips.reserve(computeHiZMipCount(width, height));

    const std::size_t mip0Count = static_cast<std::size_t>(width) * height;
    mips.push_back(HiZMipLevel{
        width,
        height,
        std::vector<float>(mip0Depths.begin(), mip0Depths.begin() + static_cast<std::ptrdiff_t>(mip0Count)),
    });

    while (mips.back().width > 1u || mips.back().height > 1u) {
        const HiZMipLevel& src = mips.back();
        const std::uint32_t srcW = src.width;
        const std::uint32_t srcH = src.height;
        const std::uint32_t dstW = std::max(srcW / 2u, 1u);
        const std::uint32_t dstH = std::max(srcH / 2u, 1u);
        const bool oddX = ((srcW & 1u) != 0u) && (srcW > 1u);
        const bool oddY = ((srcH & 1u) != 0u) && (srcH > 1u);
        const std::uint32_t maxX = srcW - 1u;
        const std::uint32_t maxY = srcH - 1u;

        auto load = [&](std::uint32_t x, std::uint32_t y) noexcept -> float {
            return src.depths[static_cast<std::size_t>(y) * srcW + x];
        };

        HiZMipLevel dst;
        dst.width = dstW;
        dst.height = dstH;
        dst.depths.resize(static_cast<std::size_t>(dstW) * dstH);

        for (std::uint32_t y = 0u; y < dstH; ++y) {
            const std::uint32_t y0 = std::min(y * 2u, maxY);
            const std::uint32_t y1 = std::min(y * 2u + 1u, maxY);
            const std::uint32_t y2 = std::min(y * 2u + 2u, maxY);
            for (std::uint32_t x = 0u; x < dstW; ++x) {
                const std::uint32_t x0 = std::min(x * 2u, maxX);
                const std::uint32_t x1 = std::min(x * 2u + 1u, maxX);
                const std::uint32_t x2 = std::min(x * 2u + 2u, maxX);

                float minDepth = std::min({load(x0, y0), load(x1, y0), load(x0, y1), load(x1, y1)});
                if (oddX) {
                    minDepth = std::min({minDepth, load(x2, y0), load(x2, y1)});
                }
                if (oddY) {
                    minDepth = std::min({minDepth, load(x0, y2), load(x1, y2)});
                }
                if (oddX && oddY) {
                    minDepth = std::min(minDepth, load(x2, y2));
                }
                dst.depths[static_cast<std::size_t>(y) * dstW + x] = minDepth;
            }
        }
        mips.push_back(std::move(dst));
    }

    return mips;
}

CpuCullResult cullInstancesPhase1Cpu(std::span<const CullInstance> instances,
                                     const CullConstants& constants,
                                     std::span<const HiZMipLevel> hizMips) {
    CpuCullResult result;
    result.visibility.assign(instances.size(), kCullVisibilityCulled);
    result.commands.reserve(instances.size());

    const bool doFrustum = (constants.flags & kCullFlagFrustum) != 0u;
    const std::span<const glm::vec4, 6> planes(constants.frustumPlanes);

    for (std::size_t i = 0; i < instances.size(); ++i) {
        const CullInstance& inst = instances[i];
        if (doFrustum && (inst.flags & kCullInstanceSkipFrustum) == 0u &&
            !intersectsReversedZFrustum(planes, inst.aabbCenter, inst.aabbExtents)) {
            result.visibility[i] = kCullVisibilityCulled;
            continue;
        }
        if (!isAabbVisibleInHiZCpu(inst, constants, hizMips)) {
            result.visibility[i] = kCullVisibilityOccluded;
            continue;
        }
        result.visibility[i] = kCullVisibilityPhase1;
        result.commands.push_back(makeIndirectCommand(inst));
    }

    return result;
}

std::vector<DrawIndexedIndirectCommand> cullInstancesPhase2Cpu(
    std::span<const CullInstance> instances,
    const CullConstants& constants,
    std::span<const HiZMipLevel> hizMips,
    std::span<std::uint32_t> visibility) {
    std::vector<DrawIndexedIndirectCommand> commands;
    const std::size_t count = std::min(instances.size(), visibility.size());
    const bool doFrustum = (constants.flags & kCullFlagFrustum) != 0u;
    const std::span<const glm::vec4, 6> planes(constants.frustumPlanes);

    for (std::size_t i = 0; i < count; ++i) {
        if (visibility[i] != kCullVisibilityOccluded) {
            continue;
        }
        const CullInstance& inst = instances[i];
        if (doFrustum && (inst.flags & kCullInstanceSkipFrustum) == 0u &&
            !intersectsReversedZFrustum(planes, inst.aabbCenter, inst.aabbExtents)) {
            visibility[i] = kCullVisibilityCulled;
            continue;
        }
        if (!isAabbVisibleInHiZCpu(inst, constants, hizMips)) {
            continue;
        }
        visibility[i] = kCullVisibilityPhase2;
        commands.push_back(makeIndirectCommand(inst));
    }

    return commands;
}

// ---- GpuOcclusionCuller ------------------------------------------------------------------------

GpuOcclusionCuller::GpuOcclusionCuller(RenderContext& ctx) : ctx_(ctx) {
    nvrhi::IDevice* gpu = ctx_.gpu();

    nvrhi::BindingLayoutDesc cullLayoutDesc;
    cullLayoutDesc.visibility = nvrhi::ShaderType::Compute;
    cullLayoutDesc.registerSpace = 1;
    cullLayoutDesc.registerSpaceIsDescriptorSet = true;
    cullLayoutDesc.bindingOffsets.setShaderResourceOffset(0)
        .setSamplerOffset(0)
        .setConstantBufferOffset(0)
        .setUnorderedAccessViewOffset(0);
    cullLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::ConstantBuffer(0),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(1),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(2),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(3),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(4),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(5),
        nvrhi::BindingLayoutItem::StructuredBuffer_UAV(6),
    };
    cullLayout_ = gpu->createBindingLayout(cullLayoutDesc);
    if (!cullLayout_) {
        throw rhi::VulkanError("could not create GPU culling binding layout", VK_ERROR_INITIALIZATION_FAILED);
    }

    hizBuildPipeline_ = ctx_.createComputePipeline("hiz.build.comp", "CSMain");

    nvrhi::ComputePipelineDesc desc;
    desc.bindingLayouts = {ctx_.frameLayout(), ctx_.bindless().layout(), cullLayout_};
    desc.CS = ctx_.shaders().get("gpu.cull.phase1.comp", "CSPhase1");
    cullPhase1Pipeline_ = gpu->createComputePipeline(desc);
    if (!cullPhase1Pipeline_) {
        throw rhi::VulkanError("could not create compute pipeline gpu.cull.phase1.comp", VK_ERROR_INITIALIZATION_FAILED);
    }
    desc.CS = ctx_.shaders().get("gpu.cull.phase2.comp", "CSPhase2");
    cullPhase2Pipeline_ = gpu->createComputePipeline(desc);
    if (!cullPhase2Pipeline_) {
        throw rhi::VulkanError("could not create compute pipeline gpu.cull.phase2.comp", VK_ERROR_INITIALIZATION_FAILED);
    }
}

GpuOcclusionCuller::~GpuOcclusionCuller() {
    releaseHiZResources();
}

void GpuOcclusionCuller::releaseHiZResources() {
    hizMipStorageSets_.clear();
    if (hizBindlessIndex_ != 0xffffffffu) {
        ctx_.bindless().releaseTexture(hizBindlessIndex_);
        hizBindlessIndex_ = 0xffffffffu;
    }
    hizTexture_ = nullptr;
    hizWidth_ = 0;
    hizHeight_ = 0;
    hizMipCount_ = 0;
}

void GpuOcclusionCuller::ensureHiZResources(std::uint32_t width, std::uint32_t height) {
    if (width == 0u || height == 0u) {
        return;
    }
    if (hizTexture_ && hizWidth_ == width && hizHeight_ == height) {
        return;
    }

    releaseHiZResources();

    nvrhi::IDevice* gpu = ctx_.gpu();
    const std::uint32_t mipCount = computeHiZMipCount(width, height);

    nvrhi::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.mipLevels = mipCount;
    desc.format = nvrhi::Format::R32_FLOAT;
    desc.isShaderResource = true;
    desc.isUAV = true;
    desc.initialState = nvrhi::ResourceStates::ShaderResource;
    desc.keepInitialState = true;
    desc.debugName = "HiZPyramid";

    hizTexture_ = gpu->createTexture(desc);
    if (!hizTexture_) {
        throw rhi::VulkanError("could not create Hi-Z pyramid texture", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    }
    hizBindlessIndex_ = ctx_.bindless().registerTexture(hizTexture_);
    hizWidth_ = width;
    hizHeight_ = height;
    hizMipCount_ = mipCount;

    hizMipStorageSets_.reserve(mipCount);
    for (std::uint32_t mip = 0u; mip < mipCount; ++mip) {
        nvrhi::BindingSetDesc setDesc;
        setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(
            0, hizTexture_, nvrhi::Format::R32_FLOAT, nvrhi::TextureSubresourceSet(mip, 1, 0, 1)));
        for (std::uint32_t slot = 1u; slot < RenderGraph::kStorageSlots; ++slot) {
            setDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(slot, ctx_.storageFallback()));
        }
        nvrhi::BindingSetHandle set = gpu->createBindingSet(setDesc, ctx_.storageLayout());
        if (!set) {
            throw rhi::VulkanError("could not create Hi-Z mip storage binding set", VK_ERROR_INITIALIZATION_FAILED);
        }
        hizMipStorageSets_.push_back(std::move(set));
    }
}

void GpuOcclusionCuller::ensureInstanceBuffers(std::uint32_t maxInstances) {
    const std::uint32_t needed = std::max(maxInstances, 1u);
    if (instanceBuffer_ && instanceCapacity_ >= needed) {
        return;
    }

    std::uint32_t newCapacity = std::max(instanceCapacity_, 256u);
    while (newCapacity < needed) {
        newCapacity *= 2u;
    }

    nvrhi::IDevice* gpu = ctx_.gpu();
    instanceBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(CullInstance), sizeof(CullInstance),
        false, false, true, nvrhi::ResourceStates::UnorderedAccess, "GpuCullInstances");
    visibilityBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(std::uint32_t), sizeof(std::uint32_t),
        false, false, true, nvrhi::ResourceStates::UnorderedAccess, "GpuCullVisibility");
    phase1IndirectBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(DrawIndexedIndirectCommand),
        sizeof(DrawIndexedIndirectCommand), false, true, true,
        nvrhi::ResourceStates::IndirectArgument, "GpuCullPhase1Indirect");
    phase2IndirectBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(DrawIndexedIndirectCommand),
        sizeof(DrawIndexedIndirectCommand), false, true, true,
        nvrhi::ResourceStates::IndirectArgument, "GpuCullPhase2Indirect");
    phase1SlotIndirectBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(DrawIndexedIndirectCommand),
        sizeof(DrawIndexedIndirectCommand), false, true, true,
        nvrhi::ResourceStates::IndirectArgument, "GpuCullPhase1SlotIndirect");
    phase2SlotIndirectBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(DrawIndexedIndirectCommand),
        sizeof(DrawIndexedIndirectCommand), false, true, true,
        nvrhi::ResourceStates::IndirectArgument, "GpuCullPhase2SlotIndirect");
    finalSlotIndirectBuffer_ = createGpuBuffer(
        gpu, static_cast<std::uint64_t>(newCapacity) * sizeof(DrawIndexedIndirectCommand),
        sizeof(DrawIndexedIndirectCommand), false, true, true,
        nvrhi::ResourceStates::IndirectArgument, "GpuCullFinalSlotIndirect");

    if (!phase1CountBuffer_) {
        phase1CountBuffer_ = createGpuBuffer(
            gpu, sizeof(std::uint32_t) * 4ull, sizeof(std::uint32_t),
            false, true, true, nvrhi::ResourceStates::IndirectArgument, "GpuCullPhase1Count");
    }
    if (!phase2CountBuffer_) {
        phase2CountBuffer_ = createGpuBuffer(
            gpu, sizeof(std::uint32_t) * 4ull, sizeof(std::uint32_t),
            false, true, true, nvrhi::ResourceStates::IndirectArgument, "GpuCullPhase2Count");
    }
    if (!phase1ConstantsBuffer_) {
        phase1ConstantsBuffer_ = createGpuBuffer(
            gpu, sizeof(CullConstants), 0, true, false, false,
            nvrhi::ResourceStates::ConstantBuffer, "GpuCullPhase1Constants");
    }
    if (!phase2ConstantsBuffer_) {
        phase2ConstantsBuffer_ = createGpuBuffer(
            gpu, sizeof(CullConstants), 0, true, false, false,
            nvrhi::ResourceStates::ConstantBuffer, "GpuCullPhase2Constants");
    }

    phase1CullSet_ = createCullBindingSet(
        gpu, cullLayout_, phase1ConstantsBuffer_, instanceBuffer_,
        phase1IndirectBuffer_, visibilityBuffer_, phase1CountBuffer_,
        phase1SlotIndirectBuffer_, finalSlotIndirectBuffer_);
    phase2CullSet_ = createCullBindingSet(
        gpu, cullLayout_, phase2ConstantsBuffer_, instanceBuffer_,
        phase2IndirectBuffer_, visibilityBuffer_, phase2CountBuffer_,
        phase2SlotIndirectBuffer_, finalSlotIndirectBuffer_);

    instanceCapacity_ = newCapacity;
}

void GpuOcclusionCuller::uploadInstances(nvrhi::ICommandList* cmd, std::span<const CullInstance> instances) {
    ensureInstanceBuffers(static_cast<std::uint32_t>(instances.size()));
    if (!instances.empty()) {
        cmd->writeBuffer(instanceBuffer_, instances.data(), instances.size_bytes());
    }
}

void GpuOcclusionCuller::buildHiZ(nvrhi::ICommandList* cmd,
                                  nvrhi::IBindingSet* frameBindings,
                                  nvrhi::ITexture* depthTexture,
                                  std::uint32_t depthBindlessIndex,
                                  std::uint32_t width,
                                  std::uint32_t height) {
    if (depthTexture == nullptr || width == 0u || height == 0u) {
        return;
    }
    ensureHiZResources(width, height);

    const nvrhi::TextureDesc& depthDesc = depthTexture->getDesc();
    const std::uint32_t srcW0 = depthDesc.width;
    const std::uint32_t srcH0 = depthDesc.height;

    cmd->setTextureState(depthTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    cmd->commitBarriers();

    nvrhi::IDescriptorTable* bindlessTable = ctx_.bindless().table();

    auto dispatchMip = [&](std::uint32_t srcBindless,
                           std::uint32_t srcMip,
                           std::uint32_t srcW,
                           std::uint32_t srcH,
                           std::uint32_t dstMip,
                           std::uint32_t dstW,
                           std::uint32_t dstH,
                           std::uint32_t flags) {
        nvrhi::ComputeState state;
        state.pipeline = hizBuildPipeline_;
        state.bindings = {frameBindings, bindlessTable, hizMipStorageSets_[dstMip]};
        cmd->setComputeState(state);

        PushConstants pc;
        pc.index0 = srcBindless;
        pc.index1 = srcMip;
        pc.flags = flags;
        pc.param0 = std::bit_cast<float>(srcW);
        pc.param1 = std::bit_cast<float>(srcH);
        pc.param2 = std::bit_cast<float>(dstW);
        pc.param3 = std::bit_cast<float>(dstH);
        cmd->setPushConstants(&pc, sizeof(pc));
        cmd->dispatch((dstW + 7u) / 8u, (dstH + 7u) / 8u, 1u);
    };

    const std::uint32_t copyFlag = (srcW0 == width && srcH0 == height) ? 1u : 0u;
    dispatchMip(depthBindlessIndex, 0u, srcW0, srcH0, 0u, width, height, copyFlag);

    for (std::uint32_t mip = 1u; mip < hizMipCount_; ++mip) {
        const std::uint32_t prevMip = mip - 1u;
        const std::uint32_t srcW = std::max(width >> prevMip, 1u);
        const std::uint32_t srcH = std::max(height >> prevMip, 1u);
        const std::uint32_t dstW = std::max(width >> mip, 1u);
        const std::uint32_t dstH = std::max(height >> mip, 1u);

        cmd->setTextureState(hizTexture_, nvrhi::TextureSubresourceSet(prevMip, 1, 0, 1),
                             nvrhi::ResourceStates::ShaderResource);
        cmd->commitBarriers();

        dispatchMip(hizBindlessIndex_, prevMip, srcW, srcH, mip, dstW, dstH, 0u);
    }

    cmd->setTextureState(hizTexture_, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
    cmd->commitBarriers();
}

void GpuOcclusionCuller::dispatchCullPass(nvrhi::ICommandList* cmd,
                                          nvrhi::IComputePipeline* pipeline,
                                          nvrhi::IBuffer* constantsBuffer,
                                          nvrhi::IBuffer* countBuffer,
                                          nvrhi::IBindingSet* cullSet,
                                          nvrhi::IBindingSet* frameBindings,
                                          const glm::mat4& viewProjection,
                                          std::uint32_t instanceCount,
                                          bool hizValid,
                                          const CullConfig& config,
                                          const glm::mat4* hizViewProjection) {
    cmd->clearBufferUInt(countBuffer, 0u);
    if (instanceCount == 0u) {
        return;
    }

    const CullConstants constants = makeCullConstants(
        viewProjection, hizWidth_, hizHeight_, hizMipCount_, instanceCount,
        hizBindlessIndex_, ctx_.linearClampSampler(), config, hizValid && hizTexture_ != nullptr,
        hizViewProjection);
    cmd->writeBuffer(constantsBuffer, &constants, sizeof(constants));

    if (hizValid && hizTexture_) {
        cmd->setTextureState(hizTexture_, nvrhi::AllSubresources, nvrhi::ResourceStates::ShaderResource);
        cmd->commitBarriers();
    }

    nvrhi::ComputeState state;
    state.pipeline = pipeline;
    state.bindings = {frameBindings, ctx_.bindless().table(), cullSet};
    cmd->setComputeState(state);
    const PushConstants pc{};
    cmd->setPushConstants(&pc, sizeof(pc));
    cmd->dispatch((instanceCount + 63u) / 64u, 1u, 1u);
}

void GpuOcclusionCuller::dispatchPhase1(nvrhi::ICommandList* cmd,
                                        nvrhi::IBindingSet* frameBindings,
                                        const glm::mat4& viewProjection,
                                        std::uint32_t instanceCount,
                                        bool hizValid,
                                        const CullConfig& config,
                                        const glm::mat4* hizViewProjection) {
    ensureInstanceBuffers(std::max(instanceCount, 1u));
    dispatchCullPass(cmd, cullPhase1Pipeline_, phase1ConstantsBuffer_, phase1CountBuffer_,
                     phase1CullSet_, frameBindings, viewProjection, instanceCount, hizValid, config,
                     hizViewProjection);
    cmd->setBufferState(phase1SlotIndirectBuffer_, nvrhi::ResourceStates::IndirectArgument);
    cmd->setBufferState(finalSlotIndirectBuffer_, nvrhi::ResourceStates::IndirectArgument);
    cmd->commitBarriers();
}

void GpuOcclusionCuller::dispatchPhase2(nvrhi::ICommandList* cmd,
                                        nvrhi::IBindingSet* frameBindings,
                                        const glm::mat4& viewProjection,
                                        std::uint32_t instanceCount,
                                        const CullConfig& config) {
    ensureInstanceBuffers(std::max(instanceCount, 1u));
    dispatchCullPass(cmd, cullPhase2Pipeline_, phase2ConstantsBuffer_, phase2CountBuffer_,
                     phase2CullSet_, frameBindings, viewProjection, instanceCount,
                     hizTexture_ != nullptr, config, nullptr);
    cmd->setBufferState(phase2SlotIndirectBuffer_, nvrhi::ResourceStates::IndirectArgument);
    cmd->setBufferState(finalSlotIndirectBuffer_, nvrhi::ResourceStates::IndirectArgument);
    cmd->commitBarriers();
}

} // namespace lumora::render
