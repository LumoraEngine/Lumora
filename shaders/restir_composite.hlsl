// Composites ReSTIR DI direct lighting onto the shaded opaque HDR buffer with matching fog transmittance.
//
// gPush:
//   index0 = base shaded HDR color (bindless)
//   index1 = ReSTIR direct lighting HDR output (bindless)
//   index2 = scene depth (bindless, reversed-Z: 0 = sky)
//   flags  = bit 0: volumetric fog active (volume bindless index in param2)
//   param2 = asfloat(fogVolumeBindlessIndex)
// Output (set 1, u0): RGBA16F composited HDR scene color.
#define LUMORA_COMPUTE 1
#include "common.hlsli"

[[vk::binding(0, 1)]] [[spv::format_rgba16f]] RWTexture2D<float4> gOut DX_REGISTER_SPACE(u0, space1);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW = 0u;
    uint outH = 0u;
    gOut.GetDimensions(outW, outH);
    const int2 size = int2(int(outW), int(outH));
    const int2 p = int2(id.xy);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }

    const float4 base = gTextures[gPush.index0].Load(int3(p, 0));
    const float depth = gTextures[gPush.index2].Load(int3(p, 0)).r;
    if (depth <= 0.0) {
        gOut[p] = base;
        return;
    }

    const float3 restir = gTextures[gPush.index1].Load(int3(p, 0)).rgb;
    if (any(isnan(restir)) || any(isinf(restir))) {
        gOut[p] = base;
        return;
    }

    const float2 uv = (float2(p) + 0.5) / float2(size);
    const float2 ndc = uv * 2.0 - 1.0;
    const float4 worldH = mul(gFrame.inverseViewProjection, float4(ndc, depth, 1.0));
    const float3 worldPos = worldH.xyz / (abs(worldH.w) > 1e-7 ? worldH.w : 1.0);

    float transmittance = 1.0;
    if ((gPush.flags & 1u) != 0u && gFrame.fogVolume.x != 0u) {
        const float3 delta = worldPos - gFrame.cameraPositionTime.xyz;
        const float distance = length(delta);
        const float far = gFrame.fogVolumeParams.x;
        if (distance > far) {
            const float3 dir = delta / max(distance, 1e-6);
            const float total = 1.0 - fogAmount(worldPos);
            const float nearT = 1.0 - fogAmount(gFrame.cameraPositionTime.xyz + dir * far);
            transmittance *= saturate(total / max(nearT, 1e-4));
        }
        const float4 clip = mul(gFrame.viewProjection, float4(worldPos, 1.0));
        const float2 fogUv = clip.xy / max(clip.w, 1e-6) * 0.5 + 0.5;
        const float slices = float(gFrame.fogVolume.w);
        const float boundary = sqrt(saturate(distance / far)) * slices;
        const float4 fog =
            gVolumes[asuint(gPush.param2)].SampleLevel(gSamplers[0], float3(fogUv, (boundary - 0.5) / slices), 0.0);
        transmittance *= lerp(1.0, fog.a, saturate(boundary));
    } else {
        transmittance = 1.0 - fogAmount(worldPos);
    }

    gOut[p] = float4(max(base.rgb + max(restir, 0.0.xxx) * transmittance, 0.0.xxx), base.a);
}
