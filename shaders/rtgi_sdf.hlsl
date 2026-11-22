// Ray-traced Global Illumination via 3D Signed Distance Field & Surfel Radiance Volume (compute).
//
// gPush:
//   index0 = depth texture (bindless, reversed-Z: 0 = sky)
//   index1 = 3D SDF volume (bindless gVolumes, R16_FLOAT / R32_FLOAT; negative inside, positive outside)
//   index2 = 3D surfel radiance volume (bindless gVolumes, RGBA16_FLOAT; rgb = lit radiance, a = roughness/validity)
//   flags  = bits 0..14: optional world normal texture bindless index (0x7FFFu = reconstruct from depth),
//            bit 15: packed config valid flag, bits 16..23: maxSteps, bits 24..27: raysPerPixel - 1,
//            bits 28..31: temporal frame phase
//   param0..2 = half2-packed boundsMin (x, y, z) and boundsMax (x, y, z)
//   param3    = half2-packed (bounceStrength, skyIntensity)
// Output (set 1, u0): RGBA16F indirect irradiance (rgb = irradiance, a = ambient visibility).
#define LUMORA_COMPUTE 1
#include "common.hlsli"

[[vk::binding(0, 1)]] [[spv::format_rgba16f]] RWTexture2D<float4> gGiOut DX_REGISTER_SPACE(u0, space1);

static const uint kNoNormalSlot = 0x7FFFu;

float2 unpackHalf2(float packedVal) {
    uint bits = asuint(packedVal);
    return float2(f16tof32(bits & 0xFFFFu), f16tof32(bits >> 16u));
}

float interleavedNoise(float2 p) {
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

float3 reconstructWorldPos(float2 uv, float depth) {
    float2 ndc = uv * 2.0 - 1.0;
    float4 world = mul(gFrame.inverseViewProjection, float4(ndc, depth, 1.0));
    return world.xyz / world.w;
}

float loadClampedDepth(uint depthIdx, int2 pixel, int2 depthSize) {
    int2 clamped = clamp(pixel, int2(0, 0), depthSize - 1);
    return gTextures[depthIdx].Load(int3(clamped, 0)).r;
}

float3 reconstructWorldNormal(uint depthIdx, int2 pixel, int2 depthSize, float centerDepth, float3 centerPos) {
    float dL = loadClampedDepth(depthIdx, pixel - int2(1, 0), depthSize);
    float dR = loadClampedDepth(depthIdx, pixel + int2(1, 0), depthSize);
    float dU = loadClampedDepth(depthIdx, pixel - int2(0, 1), depthSize);
    float dD = loadClampedDepth(depthIdx, pixel + int2(0, 1), depthSize);

    float2 invSize = 1.0 / float2(depthSize);
    float3 pL = dL > 0.0 ? reconstructWorldPos((float2(pixel - int2(1, 0)) + 0.5) * invSize, dL) : centerPos;
    float3 pR = dR > 0.0 ? reconstructWorldPos((float2(pixel + int2(1, 0)) + 0.5) * invSize, dR) : centerPos;
    float3 pU = dU > 0.0 ? reconstructWorldPos((float2(pixel - int2(0, 1)) + 0.5) * invSize, dU) : centerPos;
    float3 pD = dD > 0.0 ? reconstructWorldPos((float2(pixel + int2(0, 1)) + 0.5) * invSize, dD) : centerPos;

    float3 dx = float3(1.0, 0.0, 0.0);
    if (dR > 0.0 && (dL <= 0.0 || abs(dR - centerDepth) < abs(centerDepth - dL))) {
        dx = pR - centerPos;
    } else if (dL > 0.0) {
        dx = centerPos - pL;
    }

    float3 dy = float3(0.0, 1.0, 0.0);
    if (dD > 0.0 && (dU <= 0.0 || abs(dD - centerDepth) < abs(centerDepth - dU))) {
        dy = pD - centerPos;
    } else if (dU > 0.0) {
        dy = centerPos - pU;
    }

    float3 toCam = gFrame.cameraPositionTime.xyz - centerPos;
    float3 n = cross(dy, dx);
    float nLen2 = dot(n, n);
    if (nLen2 < 1e-12) {
        float camLen2 = dot(toCam, toCam);
        return camLen2 > 1e-12 ? toCam * rsqrt(camLen2) : float3(0.0, 1.0, 0.0);
    }
    n *= rsqrt(nLen2);
    if (dot(n, toCam) < 0.0) {
        n = -n;
    }
    return n;
}

bool intersectRayBox(float3 origin, float3 invDir, float3 bmin, float3 bmax, out float tEnter, out float tExit) {
    float3 t0 = (bmin - origin) * invDir;
    float3 t1 = (bmax - origin) * invDir;
    float3 tSmall = min(t0, t1);
    float3 tLarge = max(t0, t1);
    tEnter = max(tSmall.x, max(tSmall.y, tSmall.z));
    tExit = min(tLarge.x, min(tLarge.y, tLarge.z));
    return tExit > max(tEnter, 0.0);
}

float sampleSdf(uint sdfIdx, float3 worldPos, float3 bmin, float3 invExtent) {
    float3 uvw = saturate((worldPos - bmin) * invExtent);
    return gVolumes[sdfIdx].SampleLevel(gSamplers[0], uvw, 0.0).r;
}

float3 computeSdfNormal(uint sdfIdx, float3 worldPos, float3 bmin, float3 invExtent, float eps) {
    float gx = sampleSdf(sdfIdx, worldPos + float3(eps, 0.0, 0.0), bmin, invExtent) -
               sampleSdf(sdfIdx, worldPos - float3(eps, 0.0, 0.0), bmin, invExtent);
    float gy = sampleSdf(sdfIdx, worldPos + float3(0.0, eps, 0.0), bmin, invExtent) -
               sampleSdf(sdfIdx, worldPos - float3(0.0, eps, 0.0), bmin, invExtent);
    float gz = sampleSdf(sdfIdx, worldPos + float3(0.0, 0.0, eps), bmin, invExtent) -
               sampleSdf(sdfIdx, worldPos - float3(0.0, 0.0, eps), bmin, invExtent);
    float3 g = float3(gx, gy, gz);
    float len2 = dot(g, g);
    return len2 > 1e-12 ? g * rsqrt(len2) : float3(0.0, 1.0, 0.0);
}

float3 sampleSkyFallback(float3 dir) {
    if (environmentEnabled()) {
        return environmentRadiance(dir, gFrame.environmentParams.w * 0.5);
    }
    return skyGradient(dir);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint outW = 0u;
    uint outH = 0u;
    gGiOut.GetDimensions(outW, outH);
    const int2 outSize = int2(int(outW), int(outH));
    const int2 p = int2(id.xy);
    if (p.x >= outSize.x || p.y >= outSize.y) {
        return;
    }

    const uint depthIdx = gPush.index0;
    const uint sdfIdx = gPush.index1;
    const uint surfelIdx = gPush.index2;

    const int2 depthSize = max(int2(gFrame.viewportSize.xy), int2(1, 1));
    const float2 outUv = (float2(p) + 0.5) / float2(outSize);
    const int2 depthPixel = clamp(int2(outUv * float2(depthSize)), int2(0, 0), depthSize - 1);
    const float depth = gTextures[depthIdx].Load(int3(depthPixel, 0)).r;
    if (depth <= 0.0) {
        gGiOut[p] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float2 depthUv = (float2(depthPixel) + 0.5) / float2(depthSize);
    const float3 worldPos = reconstructWorldPos(depthUv, depth);

    const uint normalSlot = gPush.flags & kNoNormalSlot;
    float3 N;
    if (normalSlot != kNoNormalSlot) {
        float3 loadedN = gTextures[normalSlot].Load(int3(depthPixel, 0)).xyz;
        float len2 = dot(loadedN, loadedN);
        N = len2 > 0.01 ? loadedN * rsqrt(len2) : reconstructWorldNormal(depthIdx, depthPixel, depthSize, depth, worldPos);
    } else {
        N = reconstructWorldNormal(depthIdx, depthPixel, depthSize, depth, worldPos);
    }

    const bool hasConfig = (gPush.flags & (1u << 15u)) != 0u;
    float3 boundsMin = float3(-16.0, -4.0, -16.0);
    float3 boundsMax = float3(16.0, 12.0, 16.0);
    float bounceStrength = 1.0;
    float skyIntensity = 1.0;
    uint maxSteps = 64u;
    uint rayCount = 8u;
    uint framePhase = 0u;

    if (hasConfig) {
        float2 minXY = unpackHalf2(gPush.param0);
        float2 minZmaxX = unpackHalf2(gPush.param1);
        float2 maxYZ = unpackHalf2(gPush.param2);
        float2 bounceSky = unpackHalf2(gPush.param3);
        boundsMin = float3(minXY.x, minXY.y, minZmaxX.x);
        boundsMax = float3(minZmaxX.y, maxYZ.x, maxYZ.y);
        bounceStrength = max(bounceSky.x, 0.0);
        skyIntensity = max(bounceSky.y, 0.0);
        uint packedSteps = (gPush.flags >> 16u) & 0xFFu;
        maxSteps = packedSteps > 0u ? packedSteps : 64u;
        rayCount = ((gPush.flags >> 24u) & 0xFu) + 1u;
        framePhase = (gPush.flags >> 28u) & 0xFu;
    }

    const float3 extent = max(boundsMax - boundsMin, float3(1e-4, 1e-4, 1e-4));
    const float3 invExtent = 1.0 / extent;

    uint volW = 1u;
    uint volH = 1u;
    uint volD = 1u;
    gVolumes[sdfIdx].GetDimensions(volW, volH, volD);
    const float3 volRes = max(float3(volW, volH, volD), float3(1.0, 1.0, 1.0));
    const float3 voxelSize = extent / volRes;
    const float minVoxel = min(voxelSize.x, min(voxelSize.y, voxelSize.z));
    const float normalEps = max(0.5 * minVoxel, 1e-3);
    const float hitEps = max(0.05 * minVoxel, 1e-3);
    const float rayBias = max(0.65 * minVoxel, 0.02);
    const float maxTraceDist = length(extent);

    const float3 upAxis = abs(N.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 tangent = normalize(cross(upAxis, N));
    const float3 bitangent = cross(N, tangent);
    const float3 rayOrigin = worldPos + N * rayBias;

    const float2 jitter = float2(
        interleavedNoise(float2(p) + float(framePhase) * 5.588238),
        interleavedNoise(float2(p.yx) + float(framePhase) * 7.137921 + 13.0)
    );

    float3 radianceSum = float3(0.0, 0.0, 0.0);
    float hitSum = 0.0;

    [loop] for (uint rayIdx = 0u; rayIdx < rayCount; ++rayIdx) {
        float u1 = frac((float(rayIdx) + 0.5) / float(rayCount) + jitter.x / float(rayCount));
        float u2 = frac(float(rayIdx) * 0.61803398875 + jitter.y);
        float r = sqrt(saturate(u1));
        float phi = 2.0 * PI * u2;
        float3 localDir = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        float3 rayDir = normalize(tangent * localDir.x + bitangent * localDir.y + N * localDir.z);

        float invX = abs(rayDir.x) > 1e-6 ? 1.0 / rayDir.x : 1e6;
        float invY = abs(rayDir.y) > 1e-6 ? 1.0 / rayDir.y : 1e6;
        float invZ = abs(rayDir.z) > 1e-6 ? 1.0 / rayDir.z : 1e6;
        float3 invDir = float3(invX, invY, invZ);

        float tEnter = 0.0;
        float tExit = 0.0;
        bool hit = false;
        float hitT = 0.0;

        if (intersectRayBox(rayOrigin, invDir, boundsMin, boundsMax, tEnter, tExit)) {
            float t = max(tEnter, 0.0);
            float tEnd = min(tExit, maxTraceDist);
            float prevT = t;
            float prevD = 0.0;

            [loop] for (uint step = 0u; step < maxSteps; ++step) {
                if (t > tEnd) {
                    break;
                }
                float3 samplePos = rayOrigin + rayDir * t;
                float d = sampleSdf(sdfIdx, samplePos, boundsMin, invExtent);
                if (d <= hitEps) {
                    float3 testN = computeSdfNormal(sdfIdx, samplePos, boundsMin, invExtent, normalEps);
                    if (step == 0u && d > -0.25 * minVoxel && dot(rayDir, testN) > 0.0) {
                        prevT = t;
                        prevD = d;
                        t += max(hitEps * 2.0, 0.05 * minVoxel);
                        continue;
                    }
                    if (step > 0u && d < 0.0 && prevD > 0.0) {
                        t = prevT + (t - prevT) * (prevD / (prevD - d));
                    } else if (d > 0.0) {
                        t = min(tEnd, t + d);
                    }
                    hit = true;
                    hitT = t;
                    break;
                }
                prevT = t;
                prevD = d;
                t += max(d, hitEps);
            }
        }

        float3 skyRadiance = max(sampleSkyFallback(rayDir), float3(0.0, 0.0, 0.0)) * skyIntensity;
        if (hit) {
            float3 hitPos = rayOrigin + rayDir * hitT;
            float3 hitN = computeSdfNormal(sdfIdx, hitPos, boundsMin, invExtent, normalEps);
            float3 surfelUvw = saturate((hitPos + hitN * (0.2 * minVoxel) - boundsMin) * invExtent);
            float4 surfel = gVolumes[surfelIdx].SampleLevel(gSamplers[0], surfelUvw, 0.0);
            float3 directBounce = max(surfel.rgb, float3(0.0, 0.0, 0.0)) * bounceStrength;

            // Multi-bounce ambient fill at the hit surface + smooth fade to sky near volume edge.
            float3 secondaryAmbient = max(sampleSkyFallback(hitN), float3(0.0, 0.0, 0.0)) *
                                      (skyIntensity * bounceStrength * 0.2);
            float3 hitRadiance = directBounce + secondaryAmbient;
            float edgeFade = smoothstep(maxTraceDist * 0.85, maxTraceDist, hitT);
            radianceSum += lerp(hitRadiance, skyRadiance, edgeFade);
            hitSum += 1.0 - edgeFade;
        } else {
            radianceSum += skyRadiance;
        }
    }

    float3 indirectIrradiance = (radianceSum / float(rayCount)) * PI;
    float visibility = saturate(1.0 - hitSum / float(rayCount));
    gGiOut[p] = float4(max(indirectIrradiance, float3(0.0, 0.0, 0.0)), visibility);
}
