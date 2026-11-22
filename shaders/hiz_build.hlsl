// Reversed-Z hierarchical depth (Hi-Z) pyramid builder (compute).
//
// Builds one mip level of the reversed-Z depth pyramid (near = 1.0, far = 0.0). Because farther
// depths are smaller in reversed-Z, conservative occlusion reduction takes the MINIMUM across the
// covered source texels. When a source dimension is odd (> 1), an extra boundary tap is included
// along that axis so every destination texel's UV footprint is strictly covered.
//
// gPush:
//   index0 = source texture (bindless Texture2D index: scene depth for mip 0, or Hi-Z texture)
//   index1 = source mip level (0 for scene depth, or mip - 1 for Hi-Z chain)
//   flags  = bit 0: 1:1 copy (used for mip 0 when srcSize == dstSize)
//   param0 = asfloat(srcWidth)
//   param1 = asfloat(srcHeight)
//   param2 = asfloat(dstWidth)
//   param3 = asfloat(dstHeight)
// Output (set 1, u0): RWTexture2D<float> (R32_FLOAT) target mip level.
#define LUMORA_COMPUTE 1
#include "common.hlsli"

[[vk::binding(0, 1)]] [[spv::format_r32f]] RWTexture2D<float> gDstMip DX_REGISTER_SPACE(u0, space1);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    const uint2 dstSize = uint2(asuint(gPush.param2), asuint(gPush.param3));
    if (id.x >= dstSize.x || id.y >= dstSize.y) {
        return;
    }

    const uint2 srcSize = uint2(asuint(gPush.param0), asuint(gPush.param1));
    const int srcMip = int(gPush.index1);
    const int2 p = int2(id.xy);
    const int2 maxCoord = max(int2(srcSize) - 1, int2(0, 0));
    Texture2D srcTex = gTextures[gPush.index0];

    // Direct 1:1 copy when building mip 0 at full depth resolution.
    if ((gPush.flags & 1u) != 0u || (srcSize.x == dstSize.x && srcSize.y == dstSize.y)) {
        const int2 sp = min(p, maxCoord);
        gDstMip[p] = srcTex.Load(int3(sp, srcMip)).r;
        return;
    }

    const int2 base = p * 2;
    const int x0 = min(base.x, maxCoord.x);
    const int x1 = min(base.x + 1, maxCoord.x);
    const int y0 = min(base.y, maxCoord.y);
    const int y1 = min(base.y + 1, maxCoord.y);

    const float d00 = srcTex.Load(int3(x0, y0, srcMip)).r;
    const float d10 = srcTex.Load(int3(x1, y0, srcMip)).r;
    const float d01 = srcTex.Load(int3(x0, y1, srcMip)).r;
    const float d11 = srcTex.Load(int3(x1, y1, srcMip)).r;
    float minDepth = min(min(d00, d10), min(d01, d11));

    // When a source dimension is odd (> 1) and dst = floor(src / 2), each destination texel's
    // normalized UV footprint overlaps 3 source texels along that axis (2*p, 2*p+1, 2*p+2).
    const bool oddX = ((srcSize.x & 1u) != 0u) && (srcSize.x > 1u);
    const bool oddY = ((srcSize.y & 1u) != 0u) && (srcSize.y > 1u);

    if (oddX) {
        const int x2 = min(base.x + 2, maxCoord.x);
        const float d20 = srcTex.Load(int3(x2, y0, srcMip)).r;
        const float d21 = srcTex.Load(int3(x2, y1, srcMip)).r;
        minDepth = min(minDepth, min(d20, d21));
    }
    if (oddY) {
        const int y2 = min(base.y + 2, maxCoord.y);
        const float d02 = srcTex.Load(int3(x0, y2, srcMip)).r;
        const float d12 = srcTex.Load(int3(x1, y2, srcMip)).r;
        minDepth = min(minDepth, min(d02, d12));
    }
    if (oddX && oddY) {
        const int x2 = min(base.x + 2, maxCoord.x);
        const int y2 = min(base.y + 2, maxCoord.y);
        const float d22 = srcTex.Load(int3(x2, y2, srcMip)).r;
        minDepth = min(minDepth, d22);
    }

    gDstMip[p] = minDepth;
}
