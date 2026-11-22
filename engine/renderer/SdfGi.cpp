#include "renderer/SdfGi.hpp"

#include "assets/Primitives.hpp"
#include "core/Log.hpp"
#include "renderer/RenderContext.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"
#include "scene/Components.hpp"
#include "scene/Scene.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <utility>

namespace lumora::render {

namespace {

constexpr float kPi = glm::pi<float>();
constexpr float kGoldenRatioConjugate = 0.6180339887498949f;

[[nodiscard]] std::uint32_t wangHash(std::uint32_t x) noexcept {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

[[nodiscard]] float hashToFloat01(std::uint32_t x) noexcept {
    return static_cast<float>(wangHash(x) & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

[[nodiscard]] float pointAttenuationCpu(float distance, float range) noexcept {
    const float ratio = distance / std::max(range, 1e-4f);
    const float ratio2 = ratio * ratio;
    const float window = std::clamp(1.0f - ratio2 * ratio2, 0.0f, 1.0f);
    return (window * window) / (distance * distance + 1.0f);
}

[[nodiscard]] float smoothstepCpu(float edge0, float edge1, float x) noexcept {
    const float denom = std::max(edge1 - edge0, 1e-6f);
    const float t = std::clamp((x - edge0) / denom, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

struct BoxHit {
    bool hit = false;
    float tEnter = 0.0f;
    float tExit = 0.0f;
};

[[nodiscard]] BoxHit intersectRayAabb(const glm::vec3& origin, const glm::vec3& dir,
                                        const glm::vec3& bmin, const glm::vec3& bmax) noexcept {
    float tMin = -std::numeric_limits<float>::infinity();
    float tMax = std::numeric_limits<float>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(dir[axis]) < 1e-8f) {
            if (origin[axis] < bmin[axis] || origin[axis] > bmax[axis]) {
                return {};
            }
        } else {
            const float invD = 1.0f / dir[axis];
            float t0 = (bmin[axis] - origin[axis]) * invD;
            float t1 = (bmax[axis] - origin[axis]) * invD;
            if (t0 > t1) std::swap(t0, t1);
            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
            if (tMax < tMin) return {};
        }
    }
    return {tMax > std::max(tMin, 0.0f), tMin, tMax};
}

template <typename T, typename ReadFn>
[[nodiscard]] T sampleVolumeTrilinear(const SdfVolumeData& volume, const glm::vec3& worldPos,
                                      ReadFn&& read) noexcept {
    const glm::uvec3 res = volume.config.safeResolution();
    const glm::vec3 clampedPos = glm::clamp(worldPos, volume.config.boundsMin, volume.config.boundsMax);
    const glm::vec3 uvw = (clampedPos - volume.config.boundsMin) / volume.config.extent();
    const glm::vec3 maxCoord = glm::vec3(res - glm::uvec3(1u));
    const glm::vec3 coord = glm::clamp(uvw * glm::vec3(res) - 0.5f, glm::vec3(0.0f), maxCoord);
    const glm::uvec3 i0 = glm::uvec3(glm::floor(coord));
    const glm::uvec3 i1 = glm::min(i0 + glm::uvec3(1u), res - glm::uvec3(1u));
    const glm::vec3 f = coord - glm::vec3(i0);

    auto at = [&](std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept -> T {
        const std::size_t idx = (static_cast<std::size_t>(z) * res.y + y) * res.x + x;
        return read(idx);
    };

    const T c000 = at(i0.x, i0.y, i0.z);
    const T c100 = at(i1.x, i0.y, i0.z);
    const T c010 = at(i0.x, i1.y, i0.z);
    const T c110 = at(i1.x, i1.y, i0.z);
    const T c001 = at(i0.x, i0.y, i1.z);
    const T c101 = at(i1.x, i0.y, i1.z);
    const T c011 = at(i0.x, i1.y, i1.z);
    const T c111 = at(i1.x, i1.y, i1.z);

    const T c00 = glm::mix(c000, c100, f.x);
    const T c10 = glm::mix(c010, c110, f.x);
    const T c01 = glm::mix(c001, c101, f.x);
    const T c11 = glm::mix(c011, c111, f.x);
    const T c0 = glm::mix(c00, c10, f.y);
    const T c1 = glm::mix(c01, c11, f.y);
    return glm::mix(c0, c1, f.z);
}

[[nodiscard]] glm::vec3 cosineSampleHemisphere(const glm::vec3& normal, float u1, float u2) noexcept {
    const glm::vec3 n = glm::normalize(normal);
    const glm::vec3 helper = std::abs(n.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 tangent = glm::normalize(glm::cross(helper, n));
    const glm::vec3 bitangent = glm::cross(n, tangent);
    const float r = std::sqrt(std::clamp(u1, 0.0f, 1.0f));
    const float phi = 2.0f * kPi * u2;
    const float x = r * std::cos(phi);
    const float y = r * std::sin(phi);
    const float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    return glm::normalize(tangent * x + bitangent * y + n * z);
}

[[nodiscard]] SdfRayHit traceSdfInternal(const SdfVolumeData& volume, const glm::vec3& origin,
                                         const glm::vec3& direction, float maxDistance, bool sampleSurfel) {
    SdfRayHit result{};
    if (volume.distanceField.empty()) return result;

    const float dirLen2 = glm::dot(direction, direction);
    if (dirLen2 < 1e-12f) return result;
    const glm::vec3 dir = direction * (1.0f / std::sqrt(dirLen2));

    const float maxDist = maxDistance > 0.0f ? maxDistance : volume.config.maxDistance;
    if (maxDist <= 0.0f) return result;

    const BoxHit box = intersectRayAabb(origin, dir, volume.config.boundsMin, volume.config.boundsMax);
    if (!box.hit) return result;

    const float tStart = std::max(0.0f, box.tEnter);
    const float tEnd = std::min(maxDist, box.tExit);
    if (tEnd <= tStart) return result;

    const glm::vec3 voxelSize = volume.config.voxelSize();
    const float minVoxel = std::min({voxelSize.x, voxelSize.y, voxelSize.z});
    const float hitEps = volume.config.hitThreshold > 0.0f
                             ? volume.config.hitThreshold
                             : std::max(0.02f * minVoxel, 1e-3f);
    const std::uint32_t maxSteps = std::max(volume.config.maxSteps, 1u);

    float t = tStart;
    float prevT = tStart;
    float prevD = 0.0f;

    for (std::uint32_t step = 0; step < maxSteps; ++step) {
        const glm::vec3 p = origin + dir * t;
        const float d = sampleSdfDistance(volume, p);
        result.steps = step + 1u;

        if (d <= hitEps) {
            const glm::vec3 hitN = computeSdfNormal(volume, p);
            if (step == 0u && d > -0.25f * minVoxel && glm::dot(dir, hitN) > 0.0f) {
                prevT = t;
                prevD = d;
                t += std::max(hitEps * 2.0f, 0.05f * minVoxel);
                if (t > tEnd) break;
                continue;
            }

            if (step > 0u && d < 0.0f && prevD > 0.0f) {
                t = prevT + (t - prevT) * (prevD / (prevD - d));
            } else if (d > 0.0f) {
                t = std::min(tEnd, t + d);
            }

            result.hit = true;
            result.distance = t;
            result.position = origin + dir * t;
            result.normal = computeSdfNormal(volume, result.position);
            if (sampleSurfel && !volume.surfelCache.empty()) {
                const glm::vec3 samplePos = result.position + result.normal * (0.2f * minVoxel);
                glm::vec4 surfel = sampleSdfSurfel(volume, samplePos);
                if (surfel.a <= 0.0f) {
                    surfel = sampleSdfSurfel(volume, result.position);
                }
                result.radiance = glm::vec3(surfel);
                result.roughness = surfel.a > 0.0f ? surfel.a : 0.5f;
            }
            return result;
        }

        prevT = t;
        prevD = d;
        t += std::max(d, hitEps);
        if (t > tEnd) break;
    }

    return result;
}

[[nodiscard]] glm::vec3 evaluateSkyGradient(const EnvironmentSettings& env, const glm::vec3& dir) noexcept {
    const float y = dir.y;
    const glm::vec3 horizon = env.skyHorizonColor;
    glm::vec3 sky = glm::mix(horizon, env.skyZenithColor, std::pow(std::clamp(y, 0.0f, 1.0f), 0.45f));
    sky += horizon * (0.25f * std::exp(-std::max(y, 0.0f) * 14.0f));
    const glm::vec3 below = glm::mix(horizon * 1.05f, env.groundColor, std::clamp(-y * 2.5f, 0.0f, 1.0f));
    return y >= 0.0f ? sky : below;
}

template <typename SkyFn>
[[nodiscard]] glm::vec3 evaluateSdfGiPointImpl(const SdfVolumeData& volume, const glm::vec3& position,
                                               const glm::vec3& normal, SkyFn&& skyFn,
                                               std::uint32_t sampleCount, std::uint32_t seed) {
    const float nLen2 = glm::dot(normal, normal);
    if (nLen2 < 1e-12f) return glm::vec3(0.0f);
    const glm::vec3 n = normal * (1.0f / std::sqrt(nLen2));

    const std::uint32_t rays = sampleCount > 0u ? sampleCount : std::max(volume.config.raysPerPoint, 1u);
    const glm::vec3 voxelSize = volume.config.voxelSize();
    const float minVoxel = std::min({voxelSize.x, voxelSize.y, voxelSize.z});
    const glm::vec3 origin = position + n * std::max(0.5f * minVoxel, 0.02f);

    const float u1Jitter = hashToFloat01(seed * 1973u + 17u) / static_cast<float>(rays);
    const float u2Jitter = hashToFloat01(seed * 9277u + 131u);

    glm::vec3 radianceSum(0.0f);
    for (std::uint32_t i = 0; i < rays; ++i) {
        const float u1 = std::fmod((static_cast<float>(i) + 0.5f) / static_cast<float>(rays) + u1Jitter, 1.0f);
        const float u2 = std::fmod(static_cast<float>(i) * kGoldenRatioConjugate + u2Jitter, 1.0f);
        const glm::vec3 dir = cosineSampleHemisphere(n, u1, u2);
        const SdfRayHit hit = traceSdfInternal(volume, origin, dir, volume.config.maxDistance, true);
        if (hit.hit) {
            radianceSum += glm::max(hit.radiance, glm::vec3(0.0f)) * volume.config.bounceStrength;
        } else {
            radianceSum += glm::max(skyFn(dir), glm::vec3(0.0f)) * volume.config.skyIntensity;
        }
    }

    // Cosine-weighted PDF is cos(theta)/pi, so irradiance = pi * mean(L_i).
    return (radianceSum / static_cast<float>(rays)) * kPi;
}

void injectLightsInternal(SdfVolumeData& volume,
                          std::span<const std::pair<glm::vec3, glm::vec3>> directionalLights,
                          std::span<const GpuLight> localLights,
                          const glm::vec3& ambientColor) {
    const glm::uvec3 res = volume.config.safeResolution();
    const std::size_t count = volume.config.voxelCount();
    if (volume.distanceField.size() != count) return;
    volume.surfelCache.resize(count, glm::vec4(0.0f));

    const glm::vec3 voxelSize = volume.config.voxelSize();
    const float maxVoxel = std::max({voxelSize.x, voxelSize.y, voxelSize.z});
    const float band = std::max(2.0f * glm::length(voxelSize), 0.25f);
    const float shadowBias = std::max(0.75f * maxVoxel, 0.04f);

    const bool hasAlbedo = volume.surfelAlbedo.size() == count;
    const bool hasEmissive = volume.surfelEmissive.size() == count;
    const bool hasNormals = volume.surfelNormals.size() == count;

    for (std::uint32_t z = 0; z < res.z; ++z) {
        for (std::uint32_t y = 0; y < res.y; ++y) {
            for (std::uint32_t x = 0; x < res.x; ++x) {
                const std::size_t idx = (static_cast<std::size_t>(z) * res.y + y) * res.x + x;
                const float d = volume.distanceField[idx];
                const bool valid = hasEmissive ? (volume.surfelEmissive[idx].a > 0.0f) : (d <= band);
                if (!valid) {
                    volume.surfelCache[idx] = glm::vec4(0.0f);
                    continue;
                }

                const glm::vec3 albedo = hasAlbedo ? glm::max(glm::vec3(volume.surfelAlbedo[idx]), glm::vec3(0.0f))
                                                   : glm::vec3(0.8f);
                const float roughness = hasAlbedo ? std::clamp(volume.surfelAlbedo[idx].a, 0.05f, 1.0f) : 0.5f;
                const glm::vec3 emissive = hasEmissive ? glm::max(glm::vec3(volume.surfelEmissive[idx]), glm::vec3(0.0f))
                                                       : glm::vec3(0.0f);
                const glm::vec3 worldPos = volume.voxelWorldPosition(x, y, z);
                const glm::vec3 normal = hasNormals ? volume.surfelNormals[idx] : computeSdfNormal(volume, worldPos);
                const glm::vec3 surfacePos = worldPos - normal * d;

                glm::vec3 directIrradiance(0.0f);

                for (const auto& [dir, color] : directionalLights) {
                    const float len2 = glm::dot(dir, dir);
                    if (len2 < 1e-12f || glm::dot(color, color) <= 0.0f) continue;
                    const glm::vec3 toSun = dir * (1.0f / std::sqrt(len2));
                    const float ndotl = std::max(glm::dot(normal, toSun), 0.0f);
                    if (ndotl <= 0.0f) continue;

                    const glm::vec3 shadowOrigin = surfacePos + normal * shadowBias + toSun * (0.25f * maxVoxel);
                    const SdfRayHit shadowHit = traceSdfInternal(volume, shadowOrigin, toSun, volume.config.maxDistance, false);
                    if (!shadowHit.hit) {
                        directIrradiance += glm::max(color, glm::vec3(0.0f)) * ndotl;
                    }
                }

                for (const GpuLight& light : localLights) {
                    const glm::vec3 lightPos(light.positionRange);
                    const float range = light.positionRange.w;
                    const glm::vec3 toLight = lightPos - surfacePos;
                    const float dist = glm::length(toLight);
                    if (dist <= 1e-4f || (range > 0.0f && dist >= range)) continue;
                    const glm::vec3 l = toLight / dist;
                    const float ndotl = std::max(glm::dot(normal, l), 0.0f);
                    if (ndotl <= 0.0f) continue;

                    float atten = pointAttenuationCpu(dist, range);
                    if (static_cast<int>(light.directionType.w) == static_cast<int>(LightType::Spot)) {
                        const glm::vec3 spotDir = glm::normalize(glm::vec3(light.directionType));
                        const float cosAngle = glm::dot(-l, spotDir);
                        atten *= smoothstepCpu(light.spot.x, light.spot.y, cosAngle);
                    }
                    if (atten <= 0.0f) continue;

                    const glm::vec3 shadowOrigin = surfacePos + normal * shadowBias + l * (0.25f * maxVoxel);
                    const float shadowDist = dist - shadowBias - 0.25f * maxVoxel;
                    bool occluded = false;
                    if (shadowDist > 1e-3f) {
                        occluded = traceSdfInternal(volume, shadowOrigin, l, shadowDist, false).hit;
                    }
                    if (!occluded) {
                        const glm::vec3 lightRadiance = glm::max(glm::vec3(light.colorIntensity), glm::vec3(0.0f)) *
                                                        std::max(light.colorIntensity.w, 0.0f) * atten;
                        directIrradiance += lightRadiance * ndotl;
                    }
                }

                const glm::vec3 radiance = directIrradiance * (albedo / kPi) +
                                           glm::max(ambientColor, glm::vec3(0.0f)) * albedo +
                                           emissive;
                volume.surfelCache[idx] = glm::vec4(radiance, roughness);
            }
        }
    }
}

[[nodiscard]] std::uint32_t packHalf2(float a, float b) noexcept {
    const std::uint32_t lo = glm::packHalf1x16(std::clamp(a, -60000.0f, 60000.0f));
    const std::uint32_t hi = glm::packHalf1x16(std::clamp(b, -60000.0f, 60000.0f));
    return lo | (hi << 16u);
}

} // namespace

float signedDistanceBox(const glm::vec3& point, const glm::vec3& boxMin, const glm::vec3& boxMax) noexcept {
    const glm::vec3 bmin = glm::min(boxMin, boxMax);
    const glm::vec3 bmax = glm::max(boxMin, boxMax);
    const glm::vec3 center = 0.5f * (bmin + bmax);
    const glm::vec3 halfExtents = 0.5f * (bmax - bmin);
    const glm::vec3 q = glm::abs(point - center) - halfExtents;
    const float outsideDist = glm::length(glm::max(q, glm::vec3(0.0f)));
    const float insideDist = std::min(std::max({q.x, q.y, q.z}), 0.0f);
    return outsideDist + insideDist;
}

glm::vec3 boxOutwardNormal(const glm::vec3& point, const glm::vec3& boxMin, const glm::vec3& boxMax) noexcept {
    const glm::vec3 bmin = glm::min(boxMin, boxMax);
    const glm::vec3 bmax = glm::max(boxMin, boxMax);
    const glm::vec3 center = 0.5f * (bmin + bmax);
    const glm::vec3 halfExtents = 0.5f * (bmax - bmin);
    const glm::vec3 delta = point - center;
    const glm::vec3 q = glm::abs(delta) - halfExtents;

    if (q.x > 0.0f || q.y > 0.0f || q.z > 0.0f) {
        const glm::vec3 closest = glm::clamp(point, bmin, bmax);
        const glm::vec3 diff = point - closest;
        const float len2 = glm::dot(diff, diff);
        if (len2 > 1e-12f) {
            return diff * (1.0f / std::sqrt(len2));
        }
    }

    int axis = 0;
    if (q.y > q.x && q.y >= q.z) axis = 1;
    else if (q.z > q.x && q.z > q.y) axis = 2;

    glm::vec3 normal(0.0f);
    normal[axis] = delta[axis] >= 0.0f ? 1.0f : -1.0f;
    return normal;
}

float sampleSdfDistance(const SdfVolumeData& volume, const glm::vec3& worldPos) noexcept {
    if (volume.distanceField.empty()) return volume.config.maxDistance;
    const glm::vec3 clamped = glm::clamp(worldPos, volume.config.boundsMin, volume.config.boundsMax);
    const float outsideOffset = glm::length(worldPos - clamped);
    const float sampled = sampleVolumeTrilinear<float>(volume, clamped, [&](std::size_t idx) noexcept {
        return volume.distanceField[idx];
    });
    return sampled + outsideOffset;
}

glm::vec4 sampleSdfSurfel(const SdfVolumeData& volume, const glm::vec3& worldPos) noexcept {
    if (volume.surfelCache.empty()) return glm::vec4(0.0f);
    return sampleVolumeTrilinear<glm::vec4>(volume, worldPos, [&](std::size_t idx) noexcept {
        return volume.surfelCache[idx];
    });
}

glm::vec3 computeSdfNormal(const SdfVolumeData& volume, const glm::vec3& worldPos) noexcept {
    if (volume.distanceField.empty()) return glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 voxel = volume.config.voxelSize();
    const float eps = std::max(0.5f * std::min({voxel.x, voxel.y, voxel.z}), 1e-3f);
    const float gx = sampleSdfDistance(volume, worldPos + glm::vec3(eps, 0.0f, 0.0f)) -
                     sampleSdfDistance(volume, worldPos - glm::vec3(eps, 0.0f, 0.0f));
    const float gy = sampleSdfDistance(volume, worldPos + glm::vec3(0.0f, eps, 0.0f)) -
                     sampleSdfDistance(volume, worldPos - glm::vec3(0.0f, eps, 0.0f));
    const float gz = sampleSdfDistance(volume, worldPos + glm::vec3(0.0f, 0.0f, eps)) -
                     sampleSdfDistance(volume, worldPos - glm::vec3(0.0f, 0.0f, eps));
    const glm::vec3 g(gx, gy, gz);
    const float len2 = glm::dot(g, g);
    return len2 > 1e-12f ? g * (1.0f / std::sqrt(len2)) : glm::vec3(0.0f, 1.0f, 0.0f);
}

SdfVolumeData buildSdfVolumeFromBoxes(const SdfGridConfig& cfg, std::span<const SdfBoxProxy> boxes) {
    SdfVolumeData volume;
    volume.config = cfg;
    volume.config.resolution = cfg.safeResolution();

    const glm::uvec3 res = volume.config.resolution;
    const std::size_t count = volume.config.voxelCount();
    volume.distanceField.assign(count, volume.config.maxDistance);
    volume.surfelCache.assign(count, glm::vec4(0.0f));
    volume.surfelAlbedo.assign(count, glm::vec4(0.0f));
    volume.surfelEmissive.assign(count, glm::vec4(0.0f));
    volume.surfelNormals.assign(count, glm::vec3(0.0f, 1.0f, 0.0f));

    if (boxes.empty()) {
        return volume;
    }

    const glm::vec3 voxelSize = volume.config.voxelSize();
    const float band = std::max(2.0f * glm::length(voxelSize), 0.25f);

    for (std::uint32_t z = 0; z < res.z; ++z) {
        for (std::uint32_t y = 0; y < res.y; ++y) {
            for (std::uint32_t x = 0; x < res.x; ++x) {
                const glm::vec3 p = volume.voxelWorldPosition(x, y, z);
                float bestDist = std::numeric_limits<float>::max();
                std::size_t bestBoxIdx = 0;
                for (std::size_t b = 0; b < boxes.size(); ++b) {
                    const float d = signedDistanceBox(p, boxes[b].min, boxes[b].max);
                    if (d < bestDist) {
                        bestDist = d;
                        bestBoxIdx = b;
                    }
                }

                const std::size_t idx = (static_cast<std::size_t>(z) * res.y + y) * res.x + x;
                const SdfBoxProxy& box = boxes[bestBoxIdx];
                volume.distanceField[idx] = bestDist;
                volume.surfelNormals[idx] = boxOutwardNormal(p, box.min, box.max);

                if (bestDist <= band) {
                    const float roughness = std::clamp(box.roughness, 0.05f, 1.0f);
                    const glm::vec3 albedo = glm::max(box.albedo, glm::vec3(0.0f));
                    const glm::vec3 emissive = glm::max(box.emissive, glm::vec3(0.0f));
                    volume.surfelAlbedo[idx] = glm::vec4(albedo, roughness);
                    volume.surfelEmissive[idx] = glm::vec4(emissive, 1.0f);
                    volume.surfelCache[idx] = glm::vec4(emissive, roughness);
                }
            }
        }
    }

    return volume;
}

void injectDirectLightingToSurfels(SdfVolumeData& volume, const glm::vec3& sunDir, const glm::vec3& sunColor,
                                   std::span<const GpuLight> localLights, const glm::vec3& ambientColor) {
    const std::pair<glm::vec3, glm::vec3> dirLights[1] = {{sunDir, sunColor}};
    injectLightsInternal(volume, dirLights, localLights, ambientColor);
}

void propagateSdfMultiBounce(SdfVolumeData& volume, std::uint32_t bounces, std::uint32_t raysPerVoxel) {
    const glm::uvec3 res = volume.config.safeResolution();
    const std::size_t count = volume.config.voxelCount();
    if (volume.distanceField.size() != count || volume.surfelCache.size() != count) return;
    if (bounces == 0u || raysPerVoxel == 0u) return;

    const glm::vec3 voxelSize = volume.config.voxelSize();
    const float band = std::max(2.0f * glm::length(voxelSize), 0.25f);
    const bool hasAlbedo = volume.surfelAlbedo.size() == count;
    const bool hasEmissive = volume.surfelEmissive.size() == count;
    const bool hasNormals = volume.surfelNormals.size() == count;

    std::vector<glm::vec4> nextCache = volume.surfelCache;
    for (std::uint32_t bounce = 0; bounce < bounces; ++bounce) {
        for (std::uint32_t z = 0; z < res.z; ++z) {
            for (std::uint32_t y = 0; y < res.y; ++y) {
                for (std::uint32_t x = 0; x < res.x; ++x) {
                    const std::size_t idx = (static_cast<std::size_t>(z) * res.y + y) * res.x + x;
                    const float d = volume.distanceField[idx];
                    const bool valid = hasEmissive ? (volume.surfelEmissive[idx].a > 0.0f) : (d <= band);
                    if (!valid || std::abs(d) > band) continue;

                    const glm::vec3 albedo = hasAlbedo ? glm::max(glm::vec3(volume.surfelAlbedo[idx]), glm::vec3(0.0f))
                                                       : glm::vec3(0.8f);
                    const glm::vec3 worldPos = volume.voxelWorldPosition(x, y, z);
                    const glm::vec3 normal = hasNormals ? volume.surfelNormals[idx] : computeSdfNormal(volume, worldPos);
                    const glm::vec3 surfacePos = worldPos - normal * d;
                    const glm::vec3 indirectIrradiance = evaluateSdfGiPoint(
                        volume, surfacePos, normal, glm::vec3(0.0f), raysPerVoxel, static_cast<std::uint32_t>(idx) + bounce * 977u);
                    const glm::vec3 addedBounce = indirectIrradiance * (albedo / kPi);
                    nextCache[idx] = glm::vec4(glm::vec3(volume.surfelCache[idx]) + addedBounce,
                                               volume.surfelCache[idx].a);
                }
            }
        }
        volume.surfelCache = nextCache;
    }
}

SdfVolumeData buildSdfVolumeFromScene(const Scene& scene, const SdfGridConfig& cfg) {
    const entt::registry& reg = scene.registry();

    std::vector<SdfBoxProxy> boxes;
    for (auto [entity, mesh, world] : reg.view<MeshRendererComponent, WorldTransformComponent>().each()) {
        if (!mesh.visible || mesh.opacity < 0.5f) continue;

        glm::vec3 localHalfExtents(0.5f);
        if (mesh.mesh == MeshPrimitive::Plane) {
            localHalfExtents = glm::vec3(0.5f, 0.02f, 0.5f);
        }

        const glm::vec3 center(world.matrix[3]);
        const glm::mat3 m(world.matrix);
        glm::vec3 halfExtents = glm::abs(m[0]) * localHalfExtents.x +
                                glm::abs(m[1]) * localHalfExtents.y +
                                glm::abs(m[2]) * localHalfExtents.z;
        halfExtents = glm::max(halfExtents, glm::vec3(0.02f));

        SdfBoxProxy proxy;
        proxy.min = center - halfExtents;
        proxy.max = center + halfExtents;
        proxy.albedo = glm::max(mesh.baseColor, glm::vec3(0.0f));
        proxy.emissive = glm::max(mesh.emissive * mesh.emissiveStrength, glm::vec3(0.0f));
        proxy.roughness = std::clamp(mesh.roughness, 0.05f, 1.0f);
        boxes.push_back(proxy);
    }

    SdfVolumeData volume = buildSdfVolumeFromBoxes(cfg, boxes);
    if (boxes.empty()) return volume;

    std::vector<std::pair<glm::vec3, glm::vec3>> directionalLights;
    std::vector<GpuLight> localLights;
    for (auto [entity, light, world] : reg.view<LightComponent, WorldTransformComponent>().each()) {
        if (light.intensity <= 0.0f) continue;
        const glm::vec3 position(world.matrix[3]);
        const glm::vec3 travelDir = glm::normalize(glm::vec3(world.matrix * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
        if (light.type == LightType::Directional) {
            directionalLights.emplace_back(-travelDir, glm::max(light.color, glm::vec3(0.0f)) * light.intensity);
        } else {
            GpuLight gl;
            gl.positionRange = glm::vec4(position, light.range);
            gl.colorIntensity = glm::vec4(glm::max(light.color, glm::vec3(0.0f)), light.intensity);
            gl.directionType = glm::vec4(travelDir, static_cast<float>(light.type));
            if (light.type == LightType::Spot) {
                const float outer = glm::radians(std::clamp(light.spotAngleDeg, 1.0f, 89.0f));
                const float inner = outer * (1.0f - std::clamp(light.spotSoftness, 0.0f, 1.0f));
                gl.spot = glm::vec4(std::cos(outer), std::cos(inner) + 1e-4f, -1.0f, 0.0f);
            }
            localLights.push_back(gl);
        }
    }

    injectLightsInternal(volume, directionalLights, localLights, glm::vec3(0.0f));
    return volume;
}

SdfRayHit traceSdfRay(const SdfVolumeData& volume, const glm::vec3& origin,
                      const glm::vec3& direction, float maxDistance) {
    return traceSdfInternal(volume, origin, direction, maxDistance, true);
}

glm::vec3 evaluateSdfGiPoint(const SdfVolumeData& volume, const glm::vec3& position,
                               const glm::vec3& normal, const glm::vec3& skyColor,
                               std::uint32_t sampleCount, std::uint32_t seed) {
    return evaluateSdfGiPointImpl(
        volume, position, normal, [&](const glm::vec3&) noexcept { return skyColor; }, sampleCount, seed);
}

glm::vec3 evaluateSdfGiPoint(const SdfVolumeData& volume, const glm::vec3& position,
                               const glm::vec3& normal, const EnvironmentSettings& env,
                               std::uint32_t sampleCount, std::uint32_t seed) {
    return evaluateSdfGiPointImpl(
        volume, position, normal, [&](const glm::vec3& dir) noexcept { return evaluateSkyGradient(env, dir); },
        sampleCount, seed);
}

// ---- SdfGiPass ---------------------------------------------------------------------------------

SdfGiPass::SdfGiPass(RenderContext& ctx, bool loadPipeline) : ctx_(ctx) {
    if (loadPipeline) {
        ensurePipeline();
    }
}

SdfGiPass::~SdfGiPass() {
    releaseTextures();
}

void SdfGiPass::ensurePipeline() {
    if (!pipeline_) {
        pipeline_ = ctx_.createComputePipeline("rtgi.sdf.comp", "CSMain");
    }
}

void SdfGiPass::releaseTextures() {
    if (sdfIndex_ != UINT32_MAX) {
        ctx_.bindless().releaseTexture(sdfIndex_);
        sdfIndex_ = UINT32_MAX;
    }
    if (surfelIndex_ != UINT32_MAX) {
        ctx_.bindless().releaseTexture(surfelIndex_);
        surfelIndex_ = UINT32_MAX;
    }
    sdfTexture_ = nullptr;
    surfelTexture_ = nullptr;
}

void SdfGiPass::uploadVolume(const SdfVolumeData& volume) {
    const glm::uvec3 res = volume.config.safeResolution();
    const std::size_t count = volume.config.voxelCount();
    if (volume.distanceField.size() != count || volume.surfelCache.size() != count) {
        throw rhi::VulkanError("SdfGiPass::uploadVolume: volume buffer size does not match resolution",
                               VK_ERROR_INITIALIZATION_FAILED);
    }

    std::vector<std::uint16_t> sdfTexels(count);
    std::vector<std::uint16_t> surfelTexels(count * 4u);
    for (std::size_t i = 0; i < count; ++i) {
        sdfTexels[i] = glm::packHalf1x16(std::clamp(volume.distanceField[i], -60000.0f, 60000.0f));
        const glm::vec4& s = volume.surfelCache[i];
        surfelTexels[i * 4u + 0u] = glm::packHalf1x16(std::clamp(s.r, 0.0f, 60000.0f));
        surfelTexels[i * 4u + 1u] = glm::packHalf1x16(std::clamp(s.g, 0.0f, 60000.0f));
        surfelTexels[i * 4u + 2u] = glm::packHalf1x16(std::clamp(s.b, 0.0f, 60000.0f));
        surfelTexels[i * 4u + 3u] = glm::packHalf1x16(std::clamp(s.a, 0.0f, 1.0f));
    }

    const bool sameDimensions = sdfTexture_ != nullptr && surfelTexture_ != nullptr &&
                                config_.safeResolution() == res &&
                                sdfIndex_ != UINT32_MAX && surfelIndex_ != UINT32_MAX;

    if (sameDimensions) {
        const std::size_t sdfRow = static_cast<std::size_t>(res.x) * sizeof(std::uint16_t);
        const std::size_t surfelRow = static_cast<std::size_t>(res.x) * 4u * sizeof(std::uint16_t);
        ctx_.upload([&](nvrhi::ICommandList* cmd) {
            cmd->writeTexture(sdfTexture_, 0, 0, sdfTexels.data(), sdfRow, sdfRow * res.y);
            cmd->writeTexture(surfelTexture_, 0, 0, surfelTexels.data(), surfelRow, surfelRow * res.y);
        });
    } else {
        releaseTextures();
        sdfTexture_ = ctx_.createVolumeTexture(std::as_bytes(std::span(sdfTexels)), res.x, res.y, res.z,
                                               nvrhi::Format::R16_FLOAT, "SdfGiDistance");
        surfelTexture_ = ctx_.createVolumeTexture(std::as_bytes(std::span(surfelTexels)), res.x, res.y, res.z,
                                                  nvrhi::Format::RGBA16_FLOAT, "SdfGiSurfelRadiance");
        sdfIndex_ = ctx_.bindless().registerTexture(sdfTexture_);
        surfelIndex_ = ctx_.bindless().registerTexture(surfelTexture_);
    }

    config_ = volume.config;
    config_.resolution = res;
}

PushConstants SdfGiPass::makePushConstants(std::uint32_t depthIndex, std::uint32_t sdfIndex,
                                           std::uint32_t surfelIndex, const SdfGridConfig& cfg,
                                           std::uint32_t normalIndex, std::uint32_t frameIndex) noexcept {
    PushConstants pc{};
    pc.index0 = depthIndex;
    pc.index1 = sdfIndex;
    pc.index2 = surfelIndex;

    const std::uint32_t normSlot = normalIndex < kNoNormalTexture ? normalIndex : kNoNormalTexture;
    const std::uint32_t steps = std::clamp(cfg.maxSteps, 1u, 255u);
    const std::uint32_t raysMinusOne = std::clamp(cfg.raysPerPoint, 1u, 16u) - 1u;
    const std::uint32_t phase = frameIndex & 0xFu;

    pc.flags = normSlot | (1u << 15u) | (steps << 16u) | (raysMinusOne << 24u) | (phase << 28u);
    pc.param0 = std::bit_cast<float>(packHalf2(cfg.boundsMin.x, cfg.boundsMin.y));
    pc.param1 = std::bit_cast<float>(packHalf2(cfg.boundsMin.z, cfg.boundsMax.x));
    pc.param2 = std::bit_cast<float>(packHalf2(cfg.boundsMax.y, cfg.boundsMax.z));
    pc.param3 = std::bit_cast<float>(packHalf2(std::max(cfg.bounceStrength, 0.0f), std::max(cfg.skyIntensity, 0.0f)));
    return pc;
}

RGImage SdfGiPass::addPass(RenderGraph& graph, RGImage depth, std::uint32_t width, std::uint32_t height,
                           RGImage normalRoughness, std::uint32_t frameIndex) {
    const RGImage giOut = graph.createImage("SdfGiIrradiance",
                                            {nvrhi::Format::RGBA16_FLOAT, std::max(width, 1u), std::max(height, 1u), true});
    addDispatchPass(graph, giOut, depth, width, height, normalRoughness, frameIndex);
    return giOut;
}

void SdfGiPass::addDispatchPass(RenderGraph& graph, RGImage giOut, RGImage depth, std::uint32_t width,
                                std::uint32_t height, RGImage normalRoughness, std::uint32_t frameIndex) {
    ensurePipeline();
    if (!hasVolume()) {
        log::warn("SdfGiPass", "addDispatchPass called before uploadVolume(); pass will be a no-op");
    }

    graph.addComputePass(
        "SdfGi",
        [&](PassBuilder& b) {
            b.writeStorage(giOut);
            b.sample(depth);
            if (normalRoughness.valid()) {
                b.sample(normalRoughness);
            }
        },
        [pipeline = pipeline_.Get(), bindlessTable = ctx_.bindless().table(),
         sdfIdx = sdfIndex_, surfelIdx = surfelIndex_, cfg = config_,
         depth, normalRoughness, width, height, frameIndex](PassContext& ctx) {
            if (pipeline == nullptr || sdfIdx == UINT32_MAX || surfelIdx == UINT32_MAX || width == 0u || height == 0u) {
                return;
            }
            ctx.cmd->setComputeState(nvrhi::ComputeState()
                                         .setPipeline(pipeline)
                                         .addBindingSet(ctx.frameBindings())
                                         .addBindingSet(bindlessTable)
                                         .addBindingSet(ctx.storageBindings()));
            const std::uint32_t normalIdx = normalRoughness.valid() ? ctx.bindlessIndex(normalRoughness) : kNoTexture;
            const PushConstants pc = makePushConstants(ctx.bindlessIndex(depth), sdfIdx, surfelIdx, cfg,
                                                       normalIdx, frameIndex);
            ctx.cmd->setPushConstants(&pc, sizeof(pc));
            ctx.cmd->dispatch((width + 7u) / 8u, (height + 7u) / 8u, 1u);
        });
}

} // namespace lumora::render
