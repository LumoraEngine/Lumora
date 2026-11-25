// ReSTIR DI (Reservoir-based Spatio-Temporal Importance Resampling) compute passes.
//
// Two entry points compiled into restir.initial.comp and restir.spatial.comp:
//   - CSInitialTemporal: draws M uniform candidate samples from gPointLights using weighted
//     reservoir sampling (WRS), evaluates target PDF pHat at the shaded world position,
//     reprojects the previous frame's reservoir via motion vectors, validates temporal
//     depth/normal similarity, clamps prev.M <= maxTemporalM, combines with unbiased MIS,
//     and writes packed 128-bit reservoir (uint4) to u0.
//   - CSSpatialResolve: gathers spatialNeighbors random taps in disk radius, validates
//     edge-stopping depth/normal similarity, merges reservoirs with unbiased balance-heuristic
//     MIS normalization, evaluates final direct light contribution, and writes HDR output to u1.
//
// Storage bindings (set 1 / space 1):
//   u0: RWTexture2D<uint4>  gReservoirOut  (RGBA32_UINT)
//   u1: RWTexture2D<float4> gHdrOut        (RGBA16_FLOAT)
//   u2: RWTexture2D<uint4>  gPrevReservoir (RGBA32_UINT)
#define LUMORA_COMPUTE 1
#include "common.hlsli"

[[vk::binding(0, 1)]] [[spv::format_rgba32ui]] RWTexture2D<uint4> gReservoirOut DX_REGISTER_SPACE(u0, space1);
[[vk::binding(1, 1)]] [[spv::format_rgba16f]]  RWTexture2D<float4> gHdrOut DX_REGISTER_SPACE(u1, space1);
[[vk::binding(2, 1)]] [[spv::format_rgba32ui]] RWTexture2D<uint4> gPrevReservoir DX_REGISTER_SPACE(u2, space1);

static const uint RESTIR_MAX_SPATIAL_TAPS = 17u;

struct Reservoir {
    uint lightIndex;
    float weightSum;
    uint sampleCountM;
    float W;
};

Reservoir emptyReservoir() {
    Reservoir r;
    r.lightIndex = 0u;
    r.weightSum = 0.0;
    r.sampleCountM = 0u;
    r.W = 0.0;
    return r;
}

uint4 packReservoir(Reservoir r) {
    return uint4(r.lightIndex, asuint(r.weightSum), r.sampleCountM, asuint(r.W));
}

Reservoir unpackReservoir(uint4 packed) {
    Reservoir r;
    r.lightIndex = packed.x;
    r.weightSum = asfloat(packed.y);
    r.sampleCountM = packed.z;
    r.W = asfloat(packed.w);
    if (r.sampleCountM == 0u || isnan(r.weightSum) || isinf(r.weightSum) || isnan(r.W) || isinf(r.W)) {
        r = emptyReservoir();
    }
    return r;
}

uint wangHash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

uint pcgNext(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float nextFloat01(inout uint state) {
    return float(pcgNext(state) >> 8u) * (1.0 / 16777216.0);
}

uint initRng(uint2 pixel, uint frameIndex, uint salt) {
    uint s = pixel.x * 1973u + pixel.y * 9277u + frameIndex * 26699u + salt * 31847u;
    return wangHash(s) | 1u;
}

float restirPointAttenuation(float dist, float range) {
    float ratio = dist / max(range, 1e-4);
    float r2 = ratio * ratio;
    float window = saturate(1.0 - r2 * r2);
    return (window * window) / (dist * dist + 1.0);
}

float3 restirEvaluateIncidentLight(GpuLight light, float3 worldPos, float3 N) {
    float3 toLight = light.positionRange.xyz - worldPos;
    float distSq = dot(toLight, toLight);
    if (distSq <= 1e-12) {
        return float3(0.0, 0.0, 0.0);
    }
    float dist = sqrt(distSq);
    float3 L = toLight / dist;
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float attenuation = restirPointAttenuation(dist, light.positionRange.w);
    if ((int)light.directionType.w == LIGHT_SPOT) {
        float dirLenSq = dot(light.directionType.xyz, light.directionType.xyz);
        if (dirLenSq > 1e-8) {
            float3 spotDir = light.directionType.xyz * rsqrt(dirLenSq);
            float cosAngle = dot(-L, spotDir);
            float edge0 = light.spot.x;
            float edge1 = max(light.spot.y, edge0 + 1e-5);
            attenuation *= smoothstep(edge0, edge1, cosAngle);
        }
    }
    if (attenuation <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 lightRgb = max(light.colorIntensity.rgb, float3(0.0, 0.0, 0.0));
    float intensity = max(light.colorIntensity.w, 0.0);
    return lightRgb * (intensity * attenuation * NdotL);
}

#if defined(LUMORA_RESTIR_SPATIAL)
float restirLocalShadow(GpuLight light, float3 position, float3 normal, float dist) {
    int tile = (int)light.spot.z;
    uint atlas = asuint(gFrame.localShadowParams.w);
    if (tile < 0 || gFrame.localShadowParams.x <= 0.5 || atlas == NO_TEXTURE) {
        return 1.0;
    }
    if ((int)light.directionType.w != LIGHT_SPOT) {
        float3 d = position - light.positionRange.xyz;
        float3 a = abs(d);
        tile += a.x >= a.y && a.x >= a.z ? (d.x > 0.0 ? 0 : 1) : (a.y >= a.z ? (d.y > 0.0 ? 2 : 3) : (d.z > 0.0 ? 4 : 5));
    }
    float3 p = position + normal * (0.02 + dist * 0.006);
    float4 clip = mul(gFrame.localShadowMatrices[tile], float4(p, 1.0));
    if (clip.w <= 0.0) {
        return 1.0;
    }
    float3 ndc = clip.xyz / clip.w;
    if (any(abs(ndc.xy) > 1.0) || ndc.z > 1.0) {
        return 1.0;
    }
    float texel = gFrame.localShadowParams.y;
    float grid = gFrame.localShadowParams.z;
    float2 tileOrigin = float2(tile % (int)grid, tile / (int)grid) / grid;
    float2 inTile = clamp(ndc.xy * 0.5 + 0.5, texel * grid * 1.5, 1.0 - texel * grid * 1.5);
    float2 uv = tileOrigin + inTile / grid;
    uint smp = (uint)gFrame.shadowParams.y;
    float depth = ndc.z - 0.0005;
    float sum = 0.0;
    [unroll] for (int j = 0; j < 2; ++j) {
        [unroll] for (int i = 0; i < 2; ++i) {
            float2 offset = (float2(i, j) - 0.5) * texel;
            sum += gTextures[atlas].SampleCmpLevelZero(gShadowSamplers[smp], uv + offset, depth);
        }
    }
    return sum * 0.25;
}

float restirDistributionGGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float restirVisibilitySmithGGXCorrelated(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float lambdaV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float lambdaL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1e-5);
}

float3 restirFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float2 restirEnvironmentBrdf(float NdotV, float roughness) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

float restirDiffuseBurley(float NdotV, float NdotL, float LdotH, float roughness) {
    float energyBias = lerp(0.0, 0.5, roughness);
    float energyFactor = lerp(1.0, 1.0 / 1.51, roughness);
    float f90 = energyBias + 2.0 * LdotH * LdotH * roughness;
    float lightScatter = 1.0 + (f90 - 1.0) * pow(1.0 - NdotL, 5.0);
    float viewScatter = 1.0 + (f90 - 1.0) * pow(1.0 - NdotV, 5.0);
    return lightScatter * viewScatter * energyFactor / PI;
}

float3 restirEvaluatePbrLight(float3 position, float3 normal, float3 viewDir, float3 albedo, float metallic,
                              float roughness, float3 L, float3 radiance) {
    float NdotL = saturate(dot(normal, L));
    if (NdotL <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 H = normalize(viewDir + L);
    float NdotV = max(dot(normal, viewDir), 1e-4);
    float NdotH = saturate(dot(normal, H));
    float LdotH = saturate(dot(L, H));
    float alpha = roughness * roughness;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 F = restirFresnelSchlick(LdotH, F0);
    float2 envBrdf = restirEnvironmentBrdf(NdotV, roughness);
    float singleScatter = max(envBrdf.x + envBrdf.y, 1e-3);
    float3 comp = 1.0 + F0 * (1.0 / singleScatter - 1.0);
    float3 specular = restirDistributionGGX(NdotH, alpha) * restirVisibilitySmithGGXCorrelated(NdotV, NdotL, alpha) * F * comp;
    float3 diffuse = (1.0 - F) * (1.0 - metallic) * albedo * restirDiffuseBurley(NdotV, NdotL, LdotH, roughness);
    return (diffuse + specular) * radiance * NdotL;
}
#endif

float restirLuminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float restirTargetPdf(GpuLight light, float3 worldPos, float3 N) {
    return restirLuminance(restirEvaluateIncidentLight(light, worldPos, N));
}

float3 reconstructWorldPos(int2 pixel, float depth) {
    float2 uv = (float2(pixel) + 0.5) * gFrame.viewportSize.zw;
    float2 ndc = uv * 2.0 - 1.0;
    float4 worldH = mul(gFrame.inverseViewProjection, float4(ndc, depth, 1.0));
    float invW = abs(worldH.w) > 1e-7 ? (1.0 / worldH.w) : 1.0;
    return worldH.xyz * invW;
}

bool updateReservoir(inout Reservoir r, uint lightIndex, float weight, uint sampleCount, float u01) {
    float safeWeight = 0.0;
    if (weight > 0.0 && !isnan(weight) && !isinf(weight)) {
        safeWeight = weight;
    }
    r.weightSum += safeWeight;
    r.sampleCountM += sampleCount;
    if (safeWeight > 0.0 && (r.weightSum == safeWeight || u01 * r.weightSum < safeWeight)) {
        r.lightIndex = lightIndex;
        return true;
    }
    return false;
}

void finalizeReservoir(inout Reservoir r, float targetPdf) {
    if (r.sampleCountM == 0u || targetPdf <= 0.0 || r.weightSum <= 0.0 ||
        isnan(targetPdf) || isinf(targetPdf) || isnan(r.weightSum) || isinf(r.weightSum)) {
        r.W = 0.0;
        return;
    }
    r.W = r.weightSum / (float(r.sampleCountM) * targetPdf);
}

bool surfaceSimilar(float currDepth, float3 currNormal, float otherDepth, float4 otherNormalRaw,
                    float depthThreshold, float normalThreshold) {
    if (otherDepth <= 0.0 || dot(otherNormalRaw.xyz, otherNormalRaw.xyz) < 0.25) {
        return false;
    }
    float relDepthDiff = abs(currDepth - otherDepth) / max(otherDepth, 1e-6);
    if (relDepthDiff > depthThreshold) {
        return false;
    }
    float3 otherN = normalize(otherNormalRaw.xyz);
    return dot(currNormal, otherN) >= normalThreshold;
}

// gPush for CSInitialTemporal:
//   index0 = depth texture (bindless)
//   index1 = motion vector texture (bindless, or NO_TEXTURE)
//   index2 = normal + roughness texture (bindless)
//   flags  = bit0: historyValid, bit1: usePreviousViewProjection,
//            bits[2..11]: initialCandidates, bits[12..21]: maxTemporalM, bits[22..31]: frameIndex
//   param0 = asfloat(prevDepthIdx) (or NO_TEXTURE)
//   param1 = asfloat(prevNormalIdx) (or NO_TEXTURE)
//   param2 = depthThreshold
//   param3 = normalThreshold
[numthreads(8, 8, 1)]
void CSInitialTemporal(uint3 id : SV_DispatchThreadID) {
    const int2 size = int2(gFrame.viewportSize.xy);
    const int2 p = int2(id.xy);
    if (any(p >= size)) {
        return;
    }

    const uint depthIdx = gPush.index0;
    const uint motionIdx = gPush.index1;
    const uint normalIdx = gPush.index2;
    const uint lightCount = gFrame.lightGrid.w;

    const float depth = gTextures[depthIdx].Load(int3(p, 0)).r;
    const float4 normalRaw = gTextures[normalIdx].Load(int3(p, 0));
    if (depth <= 0.0 || dot(normalRaw.xyz, normalRaw.xyz) < 0.25 || lightCount == 0u) {
        gReservoirOut[p] = uint4(0u, 0u, 0u, 0u);
        return;
    }

    const float3 N = normalize(normalRaw.xyz);
    const float3 worldPos = reconstructWorldPos(p, depth);

    const bool historyValid = (gPush.flags & 1u) != 0u;
    const bool usePrevVp = (gPush.flags & 2u) != 0u;
    const uint initialCandidates = (gPush.flags >> 2u) & 0x3FFu;
    const uint maxTemporalM = (gPush.flags >> 12u) & 0x3FFu;
    const uint frameIndex = (gPush.flags >> 22u) & 0x3FFu;

    const uint prevDepthIdx = asuint(gPush.param0);
    const uint prevNormalIdx = asuint(gPush.param1);
    const float depthThreshold = max(gPush.param2, 0.0);
    const float normalThreshold = gPush.param3;

    uint rng = initRng(uint2(p), frameIndex, 1u);

    // 1. Initial candidate sampling (WRS over uniform random picks from gPointLights)
    Reservoir initRes = emptyReservoir();
    const float sourcePdf = 1.0 / float(lightCount);
    for (uint i = 0u; i < initialCandidates; ++i) {
        uint candidate = min(uint(nextFloat01(rng) * float(lightCount)), lightCount - 1u);
        float pHat = restirTargetPdf(gPointLights[candidate], worldPos, N);
        float weight = pHat / sourcePdf;
        updateReservoir(initRes, candidate, weight, 1u, nextFloat01(rng));
    }
    if (initRes.sampleCountM > 0u && initRes.weightSum > 0.0 && initRes.lightIndex < lightCount) {
        float pInit = restirTargetPdf(gPointLights[initRes.lightIndex], worldPos, N);
        finalizeReservoir(initRes, pInit);
    }

    Reservoir outRes = initRes;

    // 2. Temporal reprojection and unbiased MIS combination
    if (historyValid && maxTemporalM > 0u) {
        float2 uv = (float2(p) + 0.5) * gFrame.viewportSize.zw;
        float2 velocity = float2(0.0, 0.0);
        if (motionIdx != NO_TEXTURE) {
            velocity = gTextures[motionIdx].Load(int3(p, 0)).xy;
        }
        float2 prevUv = uv - velocity * 0.5;
        if (prevUv.x > 0.0 && prevUv.x < 1.0 && prevUv.y > 0.0 && prevUv.y < 1.0) {
            int2 prevP = clamp(int2(floor(prevUv * float2(size))), int2(0, 0), size - 1);
            uint pdIdx = prevDepthIdx != NO_TEXTURE ? prevDepthIdx : depthIdx;
            uint pnIdx = prevNormalIdx != NO_TEXTURE ? prevNormalIdx : normalIdx;
            float prevDepth = gTextures[pdIdx].Load(int3(prevP, 0)).r;
            float4 prevNormalRaw = gTextures[pnIdx].Load(int3(prevP, 0));

            float expectedPrevDepth = depth;
            if (usePrevVp) {
                float4 prevClip = mul(gFrame.previousViewProjection, float4(worldPos, 1.0));
                if (prevClip.w > 1e-5) {
                    expectedPrevDepth = prevClip.z / prevClip.w;
                }
            }

            if (surfaceSimilar(expectedPrevDepth, N, prevDepth, prevNormalRaw, depthThreshold, normalThreshold)) {
                Reservoir prevRes = unpackReservoir(gPrevReservoir[prevP]);
                prevRes.sampleCountM = min(prevRes.sampleCountM, maxTemporalM);
                if (prevRes.sampleCountM > 0u) {
                    float3 prevN = normalize(prevNormalRaw.xyz);
                    uint totalM = initRes.sampleCountM + prevRes.sampleCountM;
                    Reservoir combined = emptyReservoir();

                    if (initRes.sampleCountM > 0u && initRes.W > 0.0 && initRes.lightIndex < lightCount) {
                        GpuLight l0 = gPointLights[initRes.lightIndex];
                        float pCurr0 = restirTargetPdf(l0, worldPos, N);
                        float pPrev0 = restirTargetPdf(l0, worldPos, prevN);
                        float denom0 = float(initRes.sampleCountM) * pCurr0 + float(prevRes.sampleCountM) * pPrev0;
                        float mis0 = denom0 > 0.0 ? (float(initRes.sampleCountM) * pCurr0 / denom0) : 0.0;
                        float w0 = mis0 * float(totalM) * pCurr0 * initRes.W;
                        updateReservoir(combined, initRes.lightIndex, w0, initRes.sampleCountM, nextFloat01(rng));
                    } else {
                        combined.sampleCountM += initRes.sampleCountM;
                    }

                    if (prevRes.W > 0.0 && prevRes.lightIndex < lightCount) {
                        GpuLight l1 = gPointLights[prevRes.lightIndex];
                        float pCurr1 = restirTargetPdf(l1, worldPos, N);
                        float pPrev1 = restirTargetPdf(l1, worldPos, prevN);
                        float denom1 = float(initRes.sampleCountM) * pCurr1 + float(prevRes.sampleCountM) * pPrev1;
                        float mis1 = denom1 > 0.0 ? (float(prevRes.sampleCountM) * pPrev1 / denom1) : 0.0;
                        float w1 = mis1 * float(totalM) * pCurr1 * prevRes.W;
                        updateReservoir(combined, prevRes.lightIndex, w1, prevRes.sampleCountM, nextFloat01(rng));
                    } else {
                        combined.sampleCountM += prevRes.sampleCountM;
                    }

                    if (combined.sampleCountM > 0u && combined.weightSum > 0.0 && combined.lightIndex < lightCount) {
                        float pFinal = restirTargetPdf(gPointLights[combined.lightIndex], worldPos, N);
                        finalizeReservoir(combined, pFinal);
                    }
                    outRes = combined;
                }
            }
        }
    }

    gReservoirOut[p] = packReservoir(outRes);
}

#if defined(LUMORA_RESTIR_SPATIAL)
// gPush for CSSpatialResolve:
//   index0 = depth texture (bindless)
//   index1 = albedo texture (bindless, or NO_TEXTURE -> white)
//   index2 = normal + roughness texture (bindless)
//   flags  = bit0: disableMis (0 = unbiased balance-heuristic MIS), bits[1..7]: spatialNeighbors
//   param0 = spatialRadiusPx
//   param1 = asfloat(frameIndex)
//   param2 = depthThreshold
//   param3 = normalThreshold
[numthreads(8, 8, 1)]
void CSSpatialResolve(uint3 id : SV_DispatchThreadID) {
    const int2 size = int2(gFrame.viewportSize.xy);
    const int2 p = int2(id.xy);
    if (any(p >= size)) {
        return;
    }

    const uint depthIdx = gPush.index0;
    const uint albedoIdx = gPush.index1;
    const uint normalIdx = gPush.index2;
    const uint lightCount = gFrame.lightGrid.w;

    const float depth = gTextures[depthIdx].Load(int3(p, 0)).r;
    const float4 normalRaw = gTextures[normalIdx].Load(int3(p, 0));
    if (depth <= 0.0 || dot(normalRaw.xyz, normalRaw.xyz) < 0.25 || lightCount == 0u) {
        gHdrOut[p] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 N = normalize(normalRaw.xyz);
    const float3 worldPos = reconstructWorldPos(p, depth);
    const bool hasPbrSurface = (albedoIdx != NO_TEXTURE);
    float4 albedoMet = float4(1.0, 1.0, 1.0, 0.0);
    if (hasPbrSurface) {
        albedoMet = gTextures[albedoIdx].Load(int3(p, 0));
    }

    const bool unbiasedMis = (gPush.flags & 1u) == 0u;
    const uint neighborCount = min((gPush.flags >> 1u) & 0x7Fu, RESTIR_MAX_SPATIAL_TAPS - 1u);
    const float radiusPx = max(gPush.param0, 0.0);
    const uint frameIndex = asuint(gPush.param1);
    const float depthThreshold = max(gPush.param2, 0.0);
    const float normalThreshold = gPush.param3;

    uint rng = initRng(uint2(p), frameIndex, 2u);

    uint4 tapPacked[RESTIR_MAX_SPATIAL_TAPS];
    float3 tapPos[RESTIR_MAX_SPATIAL_TAPS];
    float3 tapNormal[RESTIR_MAX_SPATIAL_TAPS];

    tapPacked[0] = gReservoirOut[p];
    tapPos[0] = worldPos;
    tapNormal[0] = N;
    uint tapCount = 1u;

    for (uint n = 0u; n < neighborCount; ++n) {
        float angle = 2.0 * PI * nextFloat01(rng);
        float r = radiusPx * sqrt(nextFloat01(rng));
        int2 offset = int2(round(float2(cos(angle), sin(angle)) * r));
        int2 q = p + offset;
        if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y || (q.x == p.x && q.y == p.y)) {
            continue;
        }
        float qDepth = gTextures[depthIdx].Load(int3(q, 0)).r;
        float4 qNormalRaw = gTextures[normalIdx].Load(int3(q, 0));
        if (!surfaceSimilar(depth, N, qDepth, qNormalRaw, depthThreshold, normalThreshold)) {
            continue;
        }
        tapPacked[tapCount] = gReservoirOut[q];
        tapPos[tapCount] = reconstructWorldPos(q, qDepth);
        tapNormal[tapCount] = normalize(qNormalRaw.xyz);
        ++tapCount;
    }

    uint totalM = 0u;
    for (uint m = 0u; m < tapCount; ++m) {
        Reservoir rm = unpackReservoir(tapPacked[m]);
        totalM += rm.sampleCountM;
    }

    Reservoir combined = emptyReservoir();
    for (uint j = 0u; j < tapCount; ++j) {
        Reservoir src = unpackReservoir(tapPacked[j]);
        if (src.sampleCountM == 0u) {
            continue;
        }
        if (src.W <= 0.0 || src.lightIndex >= lightCount) {
            combined.sampleCountM += src.sampleCountM;
            continue;
        }
        GpuLight light = gPointLights[src.lightIndex];
        float pAtCenter = restirTargetPdf(light, worldPos, N);
        float risWeight = 0.0;
        if (pAtCenter > 0.0) {
            if (unbiasedMis) {
                float denom = 0.0;
                float num = 0.0;
                for (uint k = 0u; k < tapCount; ++k) {
                    Reservoir rk = unpackReservoir(tapPacked[k]);
                    if (rk.sampleCountM == 0u) {
                        continue;
                    }
                    float pk = restirTargetPdf(light, tapPos[k], tapNormal[k]);
                    float term = float(rk.sampleCountM) * pk;
                    denom += term;
                    if (k == j) {
                        num = term;
                    }
                }
                float misWeight = denom > 0.0 ? (num / denom) : 0.0;
                risWeight = misWeight * float(totalM) * pAtCenter * src.W;
            } else {
                risWeight = pAtCenter * src.W * float(src.sampleCountM);
            }
        }
        updateReservoir(combined, src.lightIndex, risWeight, src.sampleCountM, nextFloat01(rng));
    }

    if (combined.sampleCountM > 0u && combined.weightSum > 0.0 && combined.lightIndex < lightCount) {
        float pFinal = restirTargetPdf(gPointLights[combined.lightIndex], worldPos, N);
        finalizeReservoir(combined, pFinal);
    } else {
        combined.W = 0.0;
    }

    float3 radianceOut = float3(0.0, 0.0, 0.0);
    if (combined.W > 0.0 && combined.lightIndex < lightCount) {
        GpuLight chosenLight = gPointLights[combined.lightIndex];
        if (!hasPbrSurface) {
            radianceOut = (float3(1.0, 1.0, 1.0) / PI) * restirEvaluateIncidentLight(chosenLight, worldPos, N) * combined.W;
        } else {
            float3 toLight = chosenLight.positionRange.xyz - worldPos;
            float distSq = dot(toLight, toLight);
            if (distSq > 1e-12) {
                float dist = sqrt(distSq);
                float3 L = toLight / dist;
                float attenuation = restirPointAttenuation(dist, chosenLight.positionRange.w);
                if ((int)chosenLight.directionType.w == LIGHT_SPOT) {
                    float dirLenSq = dot(chosenLight.directionType.xyz, chosenLight.directionType.xyz);
                    if (dirLenSq > 1e-8) {
                        float3 spotDir = chosenLight.directionType.xyz * rsqrt(dirLenSq);
                        float cosAngle = dot(-L, spotDir);
                        float edge0 = chosenLight.spot.x;
                        float edge1 = max(chosenLight.spot.y, edge0 + 1e-5);
                        attenuation *= smoothstep(edge0, edge1, cosAngle);
                    }
                }
                if (attenuation > 0.0) {
                    float shadow = restirLocalShadow(chosenLight, worldPos, N, dist);
                    float3 lightRadiance = max(chosenLight.colorIntensity.rgb, float3(0.0, 0.0, 0.0)) *
                                           max(chosenLight.colorIntensity.w, 0.0) * attenuation * shadow;
                    float3 toCam = gFrame.cameraPositionTime.xyz - worldPos;
                    float camDistSq = dot(toCam, toCam);
                    float3 V = camDistSq > 1e-12 ? toCam * rsqrt(camDistSq) : N;
                    float3 albedo = max(albedoMet.rgb, float3(0.0, 0.0, 0.0));
                    float metallic = saturate(albedoMet.a);
                    float roughness = clamp(normalRaw.w, 0.04, 1.0);
                    radianceOut = restirEvaluatePbrLight(worldPos, N, V, albedo, metallic, roughness, L, lightRadiance) *
                                  combined.W;
                }
            }
        }
        if (any(isnan(radianceOut)) || any(isinf(radianceOut))) {
            radianceOut = float3(0.0, 0.0, 0.0);
        }
    }
    gHdrOut[p] = float4(max(radianceOut, float3(0.0, 0.0, 0.0)), 1.0);
}
#endif
