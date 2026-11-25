// GPU-driven two-phase Hi-Z occlusion culling & indirect draw emission (compute).
//
// Reversed-Z conventions (near = 1.0, far = 0.0), Vulkan Y-down clip space:
//   - CSPhase1: tests 6 frustum planes; if visible, projects AABB corners into UV/NDC, selects the
//     Hi-Z mip where the projected rect spans <= 2x2 texels, and loads the conservative minimum
//     occluder depth. If aabbMaxZ + depthBias < hizMinDepth, marks visibility = 2 (occluded
//     candidate for Phase 2); otherwise emits DrawIndexedIndirectCommand and marks visibility = 1.
//   - CSPhase2: re-tests instances marked visibility == 2 against the current-frame Hi-Z pyramid;
//     if now visible, emits DrawIndexedIndirectCommand into the Phase-2 buffer and marks
//     visibility = 3.
//
// Set 0: Lumora frame bindings (common.hlsli).
// Set 1: Culling resources (b0 = CullConstants, u1 = instances, u2 = indirect draws,
//        u3 = visibility mask, u4 = atomic draw count).
// Set 2: Lumora bindless textures (gTextures[], used to sample the Hi-Z pyramid).
#define LUMORA_COMPUTE 1
#include "common.hlsli"

static const uint CULL_VIS_CULLED = 0u;
static const uint CULL_VIS_PHASE1 = 1u;
static const uint CULL_VIS_OCCLUDED = 2u;
static const uint CULL_VIS_PHASE2 = 3u;

static const uint CULL_FLAG_FRUSTUM = 1u;
static const uint CULL_FLAG_OCCLUSION = 2u;
static const uint CULL_FLAG_HIZ_VALID = 4u;

static const uint CULL_INSTANCE_SKIP_OCCLUSION = 1u;
static const uint CULL_INSTANCE_SKIP_FRUSTUM = 2u;

struct CullInstance {
    float3 aabbCenter;
    uint indexCount;
    float3 aabbExtents;
    uint firstIndex;
    int vertexOffset;
    uint objectIndex;
    uint flags;
    uint reserved;
};

struct DrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

struct CullConstants {
    column_major float4x4 viewProjection;
    float4 frustumPlanes[6];
    float2 hizSize;
    float2 invHizSize;
    uint instanceCount;
    uint hizMipCount;
    uint hizTextureIndex;
    uint hizSamplerIndex;
    float nearPlane;
    float depthBias;
    uint flags;
    uint reserved;
};

[[vk::binding(0, 1)]] cbuffer CullCB DX_REGISTER_SPACE(b0, space1) { CullConstants gCull; };
[[vk::binding(1, 1)]] RWStructuredBuffer<CullInstance> gCullInstances DX_REGISTER_SPACE(u1, space1);
[[vk::binding(2, 1)]] RWStructuredBuffer<DrawIndexedIndirectCommand> gIndirectDraws DX_REGISTER_SPACE(u2, space1);
[[vk::binding(3, 1)]] RWStructuredBuffer<uint> gVisibilityMask DX_REGISTER_SPACE(u3, space1);
[[vk::binding(4, 1)]] RWStructuredBuffer<uint> gDrawCount DX_REGISTER_SPACE(u4, space1);
[[vk::binding(5, 1)]] RWStructuredBuffer<DrawIndexedIndirectCommand> gSlotIndirect DX_REGISTER_SPACE(u5, space1);
[[vk::binding(6, 1)]] RWStructuredBuffer<DrawIndexedIndirectCommand> gFinalSlotIndirect DX_REGISTER_SPACE(u6, space1);

bool testFrustumAabb(CullInstance inst) {
    if ((gCull.flags & CULL_FLAG_FRUSTUM) == 0u || (inst.flags & CULL_INSTANCE_SKIP_FRUSTUM) != 0u) {
        return true;
    }
    [unroll] for (int i = 0; i < 6; ++i) {
        float4 plane = gCull.frustumPlanes[i];
        float dist = dot(plane.xyz, inst.aabbCenter) + plane.w;
        float radius = dot(abs(plane.xyz), inst.aabbExtents);
        if (dist < -radius) {
            return false;
        }
    }
    return true;
}

bool testHiZAabb(CullInstance inst) {
    if ((gCull.flags & CULL_FLAG_OCCLUSION) == 0u || (gCull.flags & CULL_FLAG_HIZ_VALID) == 0u) {
        return true;
    }
    if ((inst.flags & CULL_INSTANCE_SKIP_OCCLUSION) != 0u) {
        return true;
    }
    if (gCull.hizMipCount == 0u || gCull.hizTextureIndex == NO_TEXTURE || gCull.hizSize.x <= 0.0 || gCull.hizSize.y <= 0.0) {
        return true;
    }

    float2 uvMin = float2(1.0, 1.0);
    float2 uvMax = float2(0.0, 0.0);
    float maxZ = 0.0;

    [unroll] for (uint c = 0u; c < 8u; ++c) {
        float sx = (c & 1u) != 0u ? 1.0 : -1.0;
        float sy = (c & 2u) != 0u ? 1.0 : -1.0;
        float sz = (c & 4u) != 0u ? 1.0 : -1.0;
        float3 corner = inst.aabbCenter + inst.aabbExtents * float3(sx, sy, sz);
        float4 clip = mul(gCull.viewProjection, float4(corner, 1.0));
        if (clip.w <= 0.0 || clip.z >= clip.w) {
            // Intersects near plane or extends behind the eye: conservatively visible.
            return true;
        }
        float3 ndc = clip.xyz / clip.w;
        float2 uv = ndc.xy * 0.5 + 0.5;
        uvMin = min(uvMin, uv);
        uvMax = max(uvMax, uv);
        maxZ = max(maxZ, saturate(ndc.z));
    }

    if (uvMax.x < 0.0 || uvMin.x > 1.0 || uvMax.y < 0.0 || uvMin.y > 1.0) {
        return true;
    }

    uvMin = saturate(uvMin);
    uvMax = saturate(uvMax);

    const float2 rectPx = (uvMax - uvMin) * gCull.hizSize;
    const float maxDim = max(rectPx.x, rectPx.y);
    int mip = int(ceil(log2(max(maxDim, 1.0))));
    mip = clamp(mip, 0, int(gCull.hizMipCount) - 1);

    const int2 baseSize = int2(gCull.hizSize);
    if (mip > 0) {
        const int prevMip = mip - 1;
        const int2 prevSize = max(baseSize >> prevMip, int2(1, 1));
        const int2 pMin = clamp(int2(floor(uvMin * float2(prevSize))), int2(0, 0), prevSize - 1);
        const int2 pMax = clamp(int2(floor(uvMax * float2(prevSize))), int2(0, 0), prevSize - 1);
        if (pMax.x - pMin.x <= 1 && pMax.y - pMin.y <= 1) {
            mip = prevMip;
        }
    }

    const int2 mipSize = max(baseSize >> mip, int2(1, 1));
    const int2 tMin = clamp(int2(floor(uvMin * float2(mipSize))), int2(0, 0), mipSize - 1);
    const int2 tMax = clamp(int2(floor(uvMax * float2(mipSize))), int2(0, 0), mipSize - 1);
    if (tMax.x - tMin.x > 1 || tMax.y - tMin.y > 1) {
        return true;
    }

    Texture2D hizTex = gTextures[gCull.hizTextureIndex];
    const float d00 = hizTex.Load(int3(tMin.x, tMin.y, mip)).r;
    const float d10 = hizTex.Load(int3(tMax.x, tMin.y, mip)).r;
    const float d01 = hizTex.Load(int3(tMin.x, tMax.y, mip)).r;
    const float d11 = hizTex.Load(int3(tMax.x, tMax.y, mip)).r;
    const float hizMinDepth = min(min(d00, d10), min(d01, d11));

    // Reversed-Z: strictly farther than the farthest occluder in footprint -> occluded.
    if (maxZ + gCull.depthBias < hizMinDepth) {
        return false;
    }
    return true;
}

uint instanceCountFor(CullInstance inst) {
    return inst.reserved > 0u ? inst.reserved : 1u;
}

void writeSlotIndirect(RWStructuredBuffer<DrawIndexedIndirectCommand> buf, uint slot, CullInstance inst, uint count) {
    DrawIndexedIndirectCommand cmd;
    cmd.indexCount = inst.indexCount;
    cmd.instanceCount = count;
    cmd.firstIndex = inst.firstIndex;
    cmd.vertexOffset = inst.vertexOffset;
    cmd.firstInstance = 0u;
    buf[slot] = cmd;
}

void emitIndirectDraw(CullInstance inst) {
    uint slot = 0u;
    InterlockedAdd(gDrawCount[0], 1u, slot);
    DrawIndexedIndirectCommand cmd;
    cmd.indexCount = inst.indexCount;
    cmd.instanceCount = instanceCountFor(inst);
    cmd.firstIndex = inst.firstIndex;
    cmd.vertexOffset = inst.vertexOffset;
    cmd.firstInstance = inst.objectIndex;
    gIndirectDraws[slot] = cmd;
}

[numthreads(64, 1, 1)]
void CSPhase1(uint3 id : SV_DispatchThreadID) {
    const uint idx = id.x;
    if (idx >= gCull.instanceCount) {
        return;
    }
    const CullInstance inst = gCullInstances[idx];
    const uint instCount = instanceCountFor(inst);
    if (!testFrustumAabb(inst)) {
        gVisibilityMask[idx] = CULL_VIS_CULLED;
        writeSlotIndirect(gSlotIndirect, idx, inst, 0u);
        writeSlotIndirect(gFinalSlotIndirect, idx, inst, 0u);
        return;
    }
    if (!testHiZAabb(inst)) {
        gVisibilityMask[idx] = CULL_VIS_OCCLUDED;
        writeSlotIndirect(gSlotIndirect, idx, inst, 0u);
        writeSlotIndirect(gFinalSlotIndirect, idx, inst, 0u);
        return;
    }
    gVisibilityMask[idx] = CULL_VIS_PHASE1;
    writeSlotIndirect(gSlotIndirect, idx, inst, instCount);
    writeSlotIndirect(gFinalSlotIndirect, idx, inst, instCount);
    emitIndirectDraw(inst);
}

[numthreads(64, 1, 1)]
void CSPhase2(uint3 id : SV_DispatchThreadID) {
    const uint idx = id.x;
    if (idx >= gCull.instanceCount) {
        return;
    }
    const CullInstance inst = gCullInstances[idx];
    if (gVisibilityMask[idx] != CULL_VIS_OCCLUDED) {
        writeSlotIndirect(gSlotIndirect, idx, inst, 0u);
        return;
    }
    if (!testFrustumAabb(inst)) {
        gVisibilityMask[idx] = CULL_VIS_CULLED;
        writeSlotIndirect(gSlotIndirect, idx, inst, 0u);
        writeSlotIndirect(gFinalSlotIndirect, idx, inst, 0u);
        return;
    }
    if (!testHiZAabb(inst)) {
        writeSlotIndirect(gSlotIndirect, idx, inst, 0u);
        writeSlotIndirect(gFinalSlotIndirect, idx, inst, 0u);
        return;
    }
    const uint instCount = instanceCountFor(inst);
    gVisibilityMask[idx] = CULL_VIS_PHASE2;
    writeSlotIndirect(gSlotIndirect, idx, inst, instCount);
    writeSlotIndirect(gFinalSlotIndirect, idx, inst, instCount);
    emitIndirectDraw(inst);
}
