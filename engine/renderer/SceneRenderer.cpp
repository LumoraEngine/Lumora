#include "renderer/SceneRenderer.hpp"

#include "renderer/SceneRendererFrame.hpp"
#include "assets/GltfImporter.hpp"

#include "assets/AssetLibrary.hpp"
#include "assets/Material.hpp"
#include "assets/MaterialGraph.hpp"
#include "assets/MeshLod.hpp"

#include "core/FileSystem.hpp"
#include "core/JobSystem.hpp"
#include "core/Log.hpp"
#include "renderer/FrameResources.hpp"
#include "renderer/GpuCulling.hpp"
#include "renderer/GpuTypes.hpp"
#include "renderer/LightGrid.hpp"
#include "renderer/LightProbes.hpp"
#include "renderer/RenderContext.hpp"
#include "renderer/RestirLighting.hpp"
#include "renderer/SdfGi.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"
#include "scene/Landscape.hpp"
#include "scene/Scene.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string>

namespace lumora::render {

Frustum Frustum::fromViewProjection(const glm::mat4& m) {
    auto row = [&](int i) { return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]); };
    Frustum f{};
    f.planes[0] = row(3) + row(0);  // left
    f.planes[1] = row(3) - row(0);  // right
    f.planes[2] = row(3) + row(1);  // top (Vulkan clip Y points down)
    f.planes[3] = row(3) - row(1);  // bottom
    f.planes[4] = row(3) - row(2);  // near (reversed-Z: z <= w); far plane is at infinity
    for (glm::vec4& p : f.planes) {
        const float len = glm::length(glm::vec3(p));
        if (len > 0.0f) p /= len;
    }
    return f;
}

bool Frustum::intersectsAabb(const glm::vec3& center, const glm::vec3& extents) const {
    for (const glm::vec4& p : planes) {
        const glm::vec3 n(p);
        const float radius = glm::dot(glm::abs(n), extents);
        if (glm::dot(n, center) + p.w < -radius) return false;
    }
    return true;
}

ShadowFit fitSunShadow(const CameraView& camera, const glm::vec3& lightDirection, float distance,
                       std::uint32_t resolution) {
    const glm::mat4 invView = glm::inverse(camera.view);
    const glm::vec3 position(invView[3]);
    const glm::vec3 right = glm::normalize(glm::vec3(invView[0]));
    const glm::vec3 up = glm::normalize(glm::vec3(invView[1]));
    const glm::vec3 forward = -glm::normalize(glm::vec3(invView[2]));
    const float tanHalf = std::tan(camera.fovYRadians * 0.5f);

    // Bounding sphere of the frustum slice [near, distance]: rotation invariant -> stable size.
    glm::vec3 corners[8];
    int n = 0;
    for (float d : {camera.nearPlane, distance}) {
        const float h = d * tanHalf;
        const float w = h * camera.aspect;
        for (float sx : {-1.0f, 1.0f}) {
            for (float sy : {-1.0f, 1.0f}) corners[n++] = position + forward * d + right * (w * sx) + up * (h * sy);
        }
    }
    glm::vec3 center(0.0f);
    for (const glm::vec3& c : corners) center += c;
    center /= 8.0f;
    float radius = 0.0f;
    for (const glm::vec3& c : corners) radius = std::max(radius, glm::length(c - center));
    radius = std::ceil(radius * 16.0f) / 16.0f;

    const glm::vec3 dir = glm::normalize(lightDirection);
    const glm::vec3 lightUp = std::abs(dir.y) > 0.99f ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);
    // Depth range symmetric around the slice centre (depth 0.5 = centre). Towards the sun it
    // includes occluders outside the view slice; away from it, receivers that a low sun spreads
    // far along its direction while they still project inside this cascade's square.
    constexpr float kCasterMargin = 60.0f;
    const float halfDepth = std::max(radius + kCasterMargin, 4.0f * radius);
    const glm::mat4 view = glm::lookAt(center - dir * halfDepth, center, lightUp);
    glm::mat4 projection = glm::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * halfDepth);

    // Snap the projected world origin to whole texels.
    const glm::mat4 vp = projection * view;
    const float halfRes = static_cast<float>(resolution) * 0.5f;
    const glm::vec4 origin = vp * glm::vec4(0, 0, 0, 1) * halfRes;
    const glm::vec2 offset = (glm::round(glm::vec2(origin)) - glm::vec2(origin)) / halfRes;
    projection[3][0] += offset.x;
    projection[3][1] += offset.y;
    return {projection * view, radius};
}

std::array<float, kShadowCascades> cascadeSplits(float nearPlane, float distance, float lambda) {
    std::array<float, kShadowCascades> splits{};
    const float n = std::max(nearPlane, 0.01f);
    const float f = std::max(distance, n * 2.0f);
    const float l = std::clamp(lambda, 0.0f, 1.0f);
    for (std::uint32_t i = 0; i < kShadowCascades; ++i) {
        const float p = static_cast<float>(i + 1) / static_cast<float>(kShadowCascades);
        splits[i] = l * n * std::pow(f / n, p) + (1.0f - l) * (n + (f - n) * p);
    }
    splits.back() = f;
    return splits;
}


SceneRenderer::SceneRenderer(RenderContext& context) : ctx_(context) {
    const nvrhi::Format hdr = ctx_.hdrFormat();
    const nvrhi::Format depth = ctx_.depthFormat();
    auto make = [&](PipelineDesc d) { return ctx_.createPipeline(d); };

    PipelineDesc sky;
    sky.name = "Sky";
    sky.vertexShader = "sky.vert";
    sky.pixelShader = "sky.frag";
    sky.colorFormats = {hdr};
    sky_ = make(sky);

    PipelineDesc opaque;
    opaque.name = "OpaquePBR";
    opaque.vertexShader = "mesh.vert";
    opaque.pixelShader = "mesh.frag";
    opaque.input = VertexInput::Mesh;
    opaque.cull = nvrhi::RasterCullMode::Back;
    opaque.depthTest = true;
    opaque.depthWrite = false;  // the depth prepass already wrote the nearest surfaces
    opaque.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
    opaque.colorFormats = {hdr};
    opaque.depthFormat = depth;
    opaque_ = make(opaque);
    opaque.name = "OpaquePBRTwoSided";
    opaque.cull = nvrhi::RasterCullMode::None;
    opaqueTwoSided_ = make(opaque);
    opaque.cull = nvrhi::RasterCullMode::Back;
    opaque.name = "OpaquePBRSkinned";
    opaque.vertexShader = "mesh.skinned.vert";
    opaque.input = VertexInput::MeshSkinned;
    opaqueSkinned_ = make(opaque);
    {
        PipelineDesc reflections = opaque;  // skinned at this point
        reflections.name = "OpaquePBRReflectionsSkinned";
        reflections.pixelShader = "mesh.reflections.frag";
        reflections.colorFormats = {hdr, kSurfaceFormat, kSurfaceFormat};
        opaqueReflectionsSkinned_ = make(reflections);
        reflections.name = "OpaquePBRReflections";
        reflections.vertexShader = "mesh.vert";
        reflections.input = VertexInput::Mesh;
        opaqueReflections_ = make(reflections);
        reflections.name = "OpaquePBRReflectionsTwoSided";
        reflections.cull = nvrhi::RasterCullMode::None;
        opaqueReflectionsTwoSided_ = make(reflections);
    }
    PipelineDesc transparent = opaque;  // depth tested against the opaque scene, never written
    transparent.name = "TransparentPBRSkinned";
    transparent.blend = BlendMode::Alpha;
    transparentSkinned_ = make(transparent);
    transparent.name = "TransparentPBR";
    transparent.vertexShader = "mesh.vert";
    transparent.input = VertexInput::Mesh;
    transparent_ = make(transparent);
    transparent.name = "TransparentPBRTwoSided";
    transparent.cull = nvrhi::RasterCullMode::None;
    transparentTwoSided_ = make(transparent);

    PipelineDesc grid;
    grid.name = "EditorGrid";
    grid.vertexShader = "grid.vert";
    grid.pixelShader = "grid.frag";
    grid.depthTest = true;
    grid.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
    grid.blend = BlendMode::Alpha;
    grid.colorFormats = {hdr};
    grid.depthFormat = depth;
    grid_ = make(grid);

    PipelineDesc particles;
    particles.name = "ParticlesAlpha";
    particles.vertexShader = "particle.vert";
    particles.pixelShader = "particle.frag";
    particles.input = VertexInput::Particle;
    particles.depthTest = true;
    particles.depthFunc = nvrhi::ComparisonFunc::GreaterOrEqual;
    particles.blend = BlendMode::Alpha;
    particles.colorFormats = {hdr};
    particles.depthFormat = depth;
    particlesAlpha_ = make(particles);
    particles.name = "ParticlesAdditive";
    particles.blend = BlendMode::Additive;
    particlesAdditive_ = make(particles);

    PipelineDesc shadow;
    shadow.name = "SunShadow";
    shadow.vertexShader = "shadow.vert";
    shadow.input = VertexInput::Mesh;
    shadow.cull = nvrhi::RasterCullMode::None;  // single-sided planes still cast shadows
    shadow.depthTest = true;
    shadow.depthWrite = true;
    shadow.depthFunc = nvrhi::ComparisonFunc::LessOrEqual;  // conventional (not reversed) depth
    shadow.depthBias = 2;
    shadow.slopeScaledDepthBias = 2.0f;
    shadow.depthFormat = depth;
    shadow_ = make(shadow);
    shadow.name = "SunShadowSkinned";
    shadow.vertexShader = "shadow.skinned.vert";
    shadow.input = VertexInput::MeshSkinnedPosition;
    shadowSkinned_ = make(shadow);
    shadow.name = "SunShadowMasked";
    shadow.vertexShader = "shadow.masked.vert";
    shadow.pixelShader = "shadow.masked.frag";
    shadow.input = VertexInput::Mesh;
    shadowMasked_ = make(shadow);
    PipelineDesc clear;
    clear.name = "VirtualShadowClear";
    clear.vertexShader = "vsm.clear.vert";
    clear.depthTest = true;
    clear.depthWrite = true;
    clear.depthFunc = nvrhi::ComparisonFunc::Always;
    clear.depthFormat = depth;
    vsmClear_ = make(clear);
    taa_ = ctx_.createComputePipeline("taa.comp");
    vsmMark_ = ctx_.createComputePipeline("vsm.mark.comp");
    motionBlur_ = ctx_.createComputePipeline("motion.blur.comp");
    contactShadows_ = ctx_.createComputePipeline("contact.shadows.comp");
    exposureHistogram_ = ctx_.createComputePipeline("exposure.histogram.comp");
    exposureAdapt_ = ctx_.createComputePipeline("exposure.adapt.comp");
    fogInject_ = ctx_.createComputePipeline("fog.inject.comp", "CSInject");
    fogIntegrate_ = ctx_.createComputePipeline("fog.integrate.comp");
    ssrTrace_ = ctx_.createComputePipeline("ssr.trace.comp", "CSTrace");
    ssrResolve_ = ctx_.createComputePipeline("ssr.resolve.comp", "CSResolve");

    PipelineDesc prepass;
    prepass.name = "DepthPrepass";
    prepass.vertexShader = "depth.vert";
    prepass.input = VertexInput::MeshPosition;
    prepass.cull = nvrhi::RasterCullMode::Back;
    prepass.depthTest = true;
    prepass.depthWrite = true;
    prepass.depthFunc = nvrhi::ComparisonFunc::Greater;
    prepass.depthFormat = depth;
    prepass.colorFormats = {kMotionFormat};  // motion vectors for TAA
    prepass.pixelShader = "depth.frag";
    depthPrepass_ = make(prepass);
    prepass.name = "DepthPrepassSkinned";
    prepass.vertexShader = "depth.skinned.vert";
    prepass.pixelShader = "depth.skinned.frag";
    prepass.input = VertexInput::MeshSkinnedPosition;
    depthPrepassSkinned_ = make(prepass);
    prepass.name = "DepthPrepassTwoSided";
    prepass.vertexShader = "depth.vert";
    prepass.pixelShader = "depth.frag";
    prepass.input = VertexInput::MeshPosition;
    prepass.cull = nvrhi::RasterCullMode::None;
    depthPrepassTwoSided_ = make(prepass);
    prepass.name = "DepthPrepassMaskedTwoSided";
    prepass.vertexShader = "depth.masked.vert";
    prepass.pixelShader = "depth.masked.frag";
    prepass.input = VertexInput::Mesh;
    depthPrepassMaskedTwoSided_ = make(prepass);
    prepass.name = "DepthPrepassMasked";
    prepass.cull = nvrhi::RasterCullMode::Back;
    depthPrepassMasked_ = make(prepass);
    prepass.name = "DepthPrepassDithered";
    prepass.vertexShader = "depth.dithered.vert";
    prepass.pixelShader = "depth.dithered.frag";
    prepass.input = VertexInput::MeshPosition;
    depthPrepassDithered_ = make(prepass);
    prepass.name = "DepthPrepassDitheredTwoSided";
    prepass.cull = nvrhi::RasterCullMode::None;
    depthPrepassDitheredTwoSided_ = make(prepass);

    for (int blur = 0; blur < 2; ++blur) {
        PipelineDesc ao;
        ao.name = blur ? "SSAOBlur" : "SSAO";
        ao.vertexShader = "ssao.vert";
        ao.pixelShader = blur ? "ssao.blur.frag" : "ssao.frag";
        ao.pixelEntry = blur ? "PSBlur" : "PSOcclusion";
        ao.colorFormats = {kAoFormat};
        (blur ? ssaoBlur_ : ssao_) = make(ao);
    }
    for (int up = 0; up < 2; ++up) {
        PipelineDesc bloom;
        bloom.name = up ? "BloomUpsample" : "BloomDownsample";
        bloom.vertexShader = "bloom.vert";
        bloom.pixelShader = up ? "bloom.up.frag" : "bloom.down.frag";
        bloom.pixelEntry = up ? "PSUpsample" : "PSDownsample";
        bloom.blend = up ? BlendMode::Additive : BlendMode::Opaque;
        bloom.colorFormats = {hdr};
        (up ? bloomUp_ : bloomDown_) = make(bloom);
    }
    PipelineDesc decal;
    decal.name = "Decals";
    decal.vertexShader = "decal.vert";
    decal.pixelShader = "decal.frag";
    decal.input = VertexInput::MeshPosition;
    decal.cull = nvrhi::RasterCullMode::Front;  // the box's far side: also drawn with the camera inside it
    decal.blend = BlendMode::Multiply;
    decal.colorFormats = {hdr};
    decal_ = make(decal);
    PipelineDesc water;
    water.name = "Water";
    water.vertexShader = "water.vert";
    water.pixelShader = "water.frag";
    water.input = VertexInput::MeshPosition;
    water.blend = BlendMode::Alpha;
    water.colorFormats = {hdr};
    water_ = make(water);

    PipelineDesc post;
    post.vertexShader = "postfx.vert";
    post.colorFormats = {hdr};
    post.name = "DepthOfField";
    post.pixelShader = "postfx.dof.frag";
    post.pixelEntry = "PSDepthOfField";
    depthOfField_ = make(post);
    post.name = "SunShafts";
    post.pixelShader = "postfx.shafts.frag";
    post.pixelEntry = "PSShafts";
    shafts_ = make(post);
    post.name = "SunShaftsAdd";
    post.pixelShader = "postfx.shaftsadd.frag";
    post.pixelEntry = "PSShaftsAdd";
    post.blend = BlendMode::Additive;
    shaftsAdd_ = make(post);
}

SceneRenderer::~SceneRenderer() {
    releaseGraphPrograms(true);
    if (hudAtlasIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(hudAtlasIndex_);
    for (const RetiredAtlas& r : retiredHudAtlases_) ctx_.bindless().releaseTexture(r.index);
    if (spriteAtlasIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(spriteAtlasIndex_);
    if (vsmPoolIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(vsmPoolIndex_);
    if (vsmPageTableIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(vsmPageTableIndex_);
    if (localAtlasIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(localAtlasIndex_);
    if (exposureIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(exposureIndex_);
    for (const std::uint32_t index : taaHistoryIndex_) {
        if (index != UINT32_MAX) ctx_.bindless().releaseTexture(index);
    }
    if (sdfGiOutputIndex_ != UINT32_MAX) ctx_.bindless().releaseTexture(sdfGiOutputIndex_);
    sdfGiOutput_ = nullptr;
    restir_.reset();
    sdfGi_.reset();
    gpuCulling_.reset();
    for (auto& [file, volume] : probeVolumes_) ctx_.bindless().releaseTexture(volume.index);
}

namespace {

template <typename T, typename Factory>
T* lazyInitPass(std::unique_ptr<T>& slot, bool& failed, const char* label, Factory&& factory) {
    if (slot) return slot.get();
    if (failed) return nullptr;
    try {
        slot = factory();
    } catch (const std::exception& e) {
        log::warn("Renderer", "{} disabled: {}", label, e.what());
        failed = true;
        return nullptr;
    }
    return slot.get();
}

} // namespace

GpuOcclusionCuller* SceneRenderer::gpuCulling() noexcept { return ensureGpuCulling(); }
SdfGiPass* SceneRenderer::sdfGi() noexcept { return ensureSdfGi(); }
RestirPass* SceneRenderer::restir() noexcept { return ensureRestir(); }

GpuOcclusionCuller* SceneRenderer::ensureGpuCulling() {
    return lazyInitPass(gpuCulling_, gpuCullingFailed_, "GPU occlusion culling",
                        [this] { return std::make_unique<GpuOcclusionCuller>(ctx_); });
}

SdfGiPass* SceneRenderer::ensureSdfGi() {
    return lazyInitPass(sdfGi_, sdfGiFailed_, "SDF global illumination",
                        [this] { return std::make_unique<SdfGiPass>(ctx_, true); });
}

RestirPass* SceneRenderer::ensureRestir() {
    return lazyInitPass(restir_, restirFailed_, "ReSTIR direct lighting",
                        [this] { return std::make_unique<RestirPass>(ctx_); });
}

void SceneRenderer::ensureSdfGiOutput(std::uint32_t width, std::uint32_t height) {
    if (width == 0u || height == 0u) return;
    if (sdfGiOutput_ && sdfGiWidth_ == width && sdfGiHeight_ == height) return;
    if (sdfGiOutputIndex_ != UINT32_MAX) {
        ctx_.bindless().releaseTexture(sdfGiOutputIndex_);
        sdfGiOutputIndex_ = UINT32_MAX;
    }
    sdfGiOutput_ = nullptr;
    nvrhi::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = nvrhi::Format::RGBA16_FLOAT;
    desc.isShaderResource = true;
    desc.isUAV = true;
    desc.debugName = "SdfGiIrradiance";
    desc.initialState = nvrhi::ResourceStates::ShaderResource;
    desc.keepInitialState = true;
    sdfGiOutput_ = ctx_.gpu()->createTexture(desc);
    if (!sdfGiOutput_) throw rhi::VulkanError("could not create SDF GI output texture", VK_ERROR_OUT_OF_DEVICE_MEMORY);
    sdfGiOutputIndex_ = ctx_.bindless().registerTexture(sdfGiOutput_);
    sdfGiWidth_ = width;
    sdfGiHeight_ = height;
}

void SceneRenderer::ensureSdfGiVolume(const Scene& scene) {
    SdfGiPass* sdfGiPass = ensureSdfGi();
    if (sdfGiPass == nullptr) return;
    if (sdfGiPass->hasVolume() && !sdfGiAutoBuilt_) return;

    const entt::registry& reg = scene.registry();
    std::uint64_t hash = 1469598103934665603ull;
    auto mix = [&](std::uint64_t v) noexcept { hash = (hash ^ v) * 1099511628211ull; };
    auto mixVec3 = [&](const glm::vec3& v) noexcept {
        mix(std::bit_cast<std::uint32_t>(v.x));
        mix(std::bit_cast<std::uint32_t>(v.y));
        mix(std::bit_cast<std::uint32_t>(v.z));
    };

    glm::vec3 sceneMin(std::numeric_limits<float>::max());
    glm::vec3 sceneMax(-std::numeric_limits<float>::max());
    bool anyMesh = false;

    for (auto [entity, mesh, world] : reg.view<MeshRendererComponent, WorldTransformComponent>().each()) {
        if (!mesh.visible || mesh.opacity < 0.5f) continue;
        anyMesh = true;
        mix(static_cast<std::uint64_t>(entt::to_integral(entity)));
        mix(static_cast<std::uint64_t>(mesh.mesh));
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                mix(std::bit_cast<std::uint32_t>(world.matrix[col][row]));
            }
        }
        mixVec3(mesh.baseColor);
        mixVec3(mesh.emissive);
        mix(std::bit_cast<std::uint32_t>(mesh.emissiveStrength));
        mix(std::bit_cast<std::uint32_t>(mesh.roughness));

        const glm::vec3 localHalf = mesh.mesh == MeshPrimitive::Plane ? glm::vec3(0.5f, 0.02f, 0.5f) : glm::vec3(0.5f);
        const glm::vec3 center(world.matrix[3]);
        const glm::mat3 m(world.matrix);
        const glm::vec3 halfExtents = glm::max(
            glm::abs(m[0]) * localHalf.x + glm::abs(m[1]) * localHalf.y + glm::abs(m[2]) * localHalf.z,
            glm::vec3(0.02f));
        sceneMin = glm::min(sceneMin, center - halfExtents);
        sceneMax = glm::max(sceneMax, center + halfExtents);
    }

    for (auto [entity, light, world] : reg.view<LightComponent, WorldTransformComponent>().each()) {
        if (light.intensity <= 0.0f) continue;
        mix(static_cast<std::uint64_t>(entt::to_integral(entity)));
        mix(static_cast<std::uint64_t>(light.type));
        mixVec3(glm::vec3(world.matrix[3]));
        mixVec3(glm::vec3(world.matrix * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
        mixVec3(light.color);
        mix(std::bit_cast<std::uint32_t>(light.intensity));
        mix(std::bit_cast<std::uint32_t>(light.range));
    }

    if (sdfGiPass->hasVolume() && sdfGiAutoBuilt_ && sdfGiSceneHash_ == hash) return;

    SdfGridConfig cfg;
    cfg.resolution = {32u, 32u, 32u};
    cfg.raysPerPoint = 8u;
    if (anyMesh) {
        const glm::vec3 center = 0.5f * (sceneMin + sceneMax);
        const glm::vec3 halfExtent =
            glm::clamp(0.5f * (sceneMax - sceneMin) + glm::vec3(2.0f), glm::vec3(4.0f), glm::vec3(64.0f));
        cfg.boundsMin = center - halfExtent;
        cfg.boundsMax = center + halfExtent;
    }
    const SdfVolumeData volume = buildSdfVolumeFromScene(scene, cfg);
    sdfGiPass->uploadVolume(volume);
    sdfGiAutoBuilt_ = true;
    sdfGiSceneHash_ = hash;
}

const SceneRenderer::ProbeVolumeGpu* SceneRenderer::probeVolume(AssetLibrary& assets, const std::string& key) {
    auto file = assets.filePath(key);
    if (!file) return nullptr;
    ProbeVolumeGpu& entry = probeVolumes_[file->string()];
    entry.lastFrame = frameCounter_;
    if (entry.failed) return nullptr;
    if (!entry.texture) {
        auto data = loadProbeVolume(*file);
        if (!data) {
            log::warn("Renderer", "light probe volume '{}': {}", key, data.error().message);
            entry.failed = true;
            return nullptr;
        }
        // 7 RGBA texels hold a probe's 27 values; coefficient texel t of every probe is block t along Z.
        const glm::uvec3 n = data->counts;
        std::vector<std::uint16_t> texels(static_cast<std::size_t>(n.x) * n.y * n.z * 7 * 4, 0);
        for (std::uint32_t z = 0; z < n.z; ++z) {
            for (std::uint32_t y = 0; y < n.y; ++y) {
                for (std::uint32_t x = 0; x < n.x; ++x) {
                    const glm::vec3* sh = &data->sh[((static_cast<std::size_t>(z) * n.y + y) * n.x + x) * 9];
                    for (std::uint32_t v = 0; v < 27; ++v) {
                        const std::uint32_t t = v / 4;
                        const std::size_t texel = ((static_cast<std::size_t>(t) * n.z + z) * n.y + y) * n.x + x;
                        texels[texel * 4 + v % 4] = glm::packHalf1x16(std::clamp(sh[v / 3][static_cast<int>(v % 3)], -60000.0f, 60000.0f));
                    }
                }
            }
        }
        entry.counts = n;
        entry.texture = ctx_.createVolumeTexture(std::as_bytes(std::span(texels)), n.x, n.y, n.z * 7,
                                                 nvrhi::Format::RGBA16_FLOAT, "LightProbeVolume");
        entry.index = ctx_.bindless().registerTexture(entry.texture);
    }
    return &entry;
}

SceneRenderer::ModelSurface SceneRenderer::modelSurface(AssetLibrary& assets, const std::string& meshAsset) {
    if (&assets != surfaceAssets_ || assets.generation() != surfaceGeneration_) {
        modelSurfaces_.clear();
        surfaceAssets_ = &assets;
        surfaceGeneration_ = assets.generation();
    }
    // By absolute path: another project (or library at the same address) has other models.
    auto canonical = assets.canonicalKey(meshAsset);
    const std::string& key = canonical ? *canonical : meshAsset;
    if (const auto it = modelSurfaces_.find(key); it != modelSurfaces_.end()) return it->second;
    ModelSurface surface;
    if (auto ref = assets.resolveMeshKey(meshAsset)) {
        if (auto model = assets.model(ref->file); model && ref->index >= 0 &&
                                                  static_cast<std::size_t>(ref->index) < (*model)->meshes.size()) {
            const int material = (*model)->meshes[static_cast<std::size_t>(ref->index)].material;
            if (material >= 0 && static_cast<std::size_t>(material) < (*model)->materials.size()) {
                const ImportedMaterial& m = (*model)->materials[static_cast<std::size_t>(material)];
                surface.alphaCutoff = importedAlphaCutoff(m);
                surface.twoSided = m.doubleSided;
                surface.transparent = m.translucent;
                surface.opacity = m.translucent ? std::clamp(m.baseColor.a, 0.0f, 1.0f) : 1.0f;
                surface.transmissive = m.transmission > 0.0f;
                surface.uvTransformU = glm::vec4(m.uvTransform.rowU(), std::max(m.ior, 1.0f));
                surface.uvTransformV = glm::vec4(m.uvTransform.rowV(), std::max(m.specularFactor, 0.0f));
                surface.specularClearcoat = glm::vec4(glm::max(m.specularColor, glm::vec3(0.0f)), std::clamp(m.clearcoat, 0.0f, 1.0f));
                surface.sheen = glm::vec4(glm::max(m.sheenColor, glm::vec3(0.0f)), std::clamp(m.sheenRoughness, 0.0f, 1.0f));
                surface.transmission = glm::vec4(std::clamp(m.transmission, 0.0f, 1.0f), std::clamp(m.clearcoatRoughness, 0.0f, 1.0f),
                                                 std::max(m.thickness, 0.0f), std::max(m.attenuationDistance, 0.0f));
                surface.attenuation = glm::vec4(glm::clamp(m.attenuationColor, glm::vec3(0.0f), glm::vec3(1.0f)),
                                                std::bit_cast<float>(m.uv1Maps));
            }
        }
    }
    modelSurfaces_.emplace(key, surface);
    return surface;
}

const SceneRenderer::GraphProgramGpu& SceneRenderer::graphProgram(const std::shared_ptr<const MaterialProgram>& program) {
    GraphProgramGpu& entry = graphPrograms_[program.get()];
    if (entry.source.lock() != program) {
        // New, or a freed program's address reused: (re)build the textures.
        if (entry.fragment) retiredGraphPrograms_.push_back(entry);
        entry = {};
        entry.source = program;
        auto upload = [&](const std::vector<glm::vec4>& code, nvrhi::TextureHandle& texture, std::uint32_t& index, const char* name) {
            if (code.empty()) return;
            const std::span<const std::byte> bytes = std::as_bytes(std::span(code));
            texture = ctx_.createTexture(std::span(&bytes, 1), static_cast<std::uint32_t>(code.size()), 1, nvrhi::Format::RGBA32_FLOAT, name);
            index = ctx_.bindless().registerTexture(texture);
        };
        upload(program->fragment, entry.fragment, entry.fragmentIndex, "Material graph (fragment)");
        upload(program->vertex, entry.vertex, entry.vertexIndex, "Material graph (vertex)");
    }
    entry.lastUsed = frameCounter_;
    return entry;
}

void SceneRenderer::releaseGraphPrograms(bool all) {
    // Programs no frame used for a while, and replaced ones, once the GPU is surely done with them.
    constexpr std::uint64_t kRetireFrames = 8;
    for (auto it = graphPrograms_.begin(); it != graphPrograms_.end();) {
        if (all || it->second.lastUsed + kRetireFrames < frameCounter_) {
            retiredGraphPrograms_.push_back(it->second);
            it = graphPrograms_.erase(it);
        } else {
            ++it;
        }
    }
    std::erase_if(retiredGraphPrograms_, [&](const GraphProgramGpu& g) {
        if (!all && g.lastUsed + 2 * kRetireFrames >= frameCounter_) return false;
        if (g.fragmentIndex != UINT32_MAX) ctx_.bindless().releaseTexture(g.fragmentIndex);
        if (g.vertexIndex != UINT32_MAX) ctx_.bindless().releaseTexture(g.vertexIndex);
        return true;
    });
}

SceneRenderer::ModelSurface SceneRenderer::materialSurface(const MaterialAsset& m) {
    ModelSurface surface;
    surface.alphaCutoff = m.alphaMode == MaterialAsset::AlphaMode::CutOut ? std::clamp(m.alphaCutoff, 0.01f, 1.0f) : 0.0f;
    surface.twoSided = m.doubleSided;
    surface.transparent = m.alphaMode == MaterialAsset::AlphaMode::Transparent;
    surface.opacity = surface.transparent ? std::clamp(m.opacity, 0.0f, 1.0f) : 1.0f;
    surface.transmissive = m.transmission > 0.0f;
    const TextureTransform uv = m.uvTransform();
    surface.uvTransformU = glm::vec4(uv.rowU(), std::max(m.ior, 1.0f));
    surface.uvTransformV = glm::vec4(uv.rowV(), std::max(m.specularFactor, 0.0f));
    surface.specularClearcoat = glm::vec4(glm::max(m.specularColor, glm::vec3(0.0f)), std::clamp(m.clearcoat, 0.0f, 1.0f));
    surface.sheen = glm::vec4(glm::max(m.sheenColor, glm::vec3(0.0f)), std::clamp(m.sheenRoughness, 0.0f, 1.0f));
    surface.transmission = glm::vec4(std::clamp(m.transmission, 0.0f, 1.0f), std::clamp(m.clearcoatRoughness, 0.0f, 1.0f),
                                     std::max(m.thickness, 0.0f), std::max(m.attenuationDistance, 0.0f));
    surface.attenuation = glm::vec4(glm::clamp(m.attenuationColor, glm::vec3(0.0f), glm::vec3(1.0f)), 0.0f);
    return surface;
}

nvrhi::IGraphicsPipeline* SceneRenderer::outputPipeline(std::unordered_map<int, nvrhi::GraphicsPipelineHandle>& cache,
                                                        const PipelineDesc& desc, nvrhi::Format format) {
    nvrhi::GraphicsPipelineHandle& pipeline = cache[static_cast<int>(format)];
    if (!pipeline) {
        PipelineDesc d = desc;
        d.colorFormats = {format};
        pipeline = ctx_.createPipeline(d);
    }
    return pipeline;
}

const GpuMesh* SceneRenderer::meshFor(const MeshRendererComponent& mesh, AssetLibrary* assets, bool block) {
    if (mesh.meshAsset.empty()) return &ctx_.meshes().primitive(mesh.mesh);
    return assets != nullptr ? ctx_.meshes().asset(*assets, mesh.meshAsset, block) : nullptr;
}

SceneDrawStats SceneRenderer::build(RenderGraph& graph, FrameResources& frame, const Scene& scene,
                                    const CameraView& camera, const RenderSettings& settings, RGImage output,
                                    nvrhi::Format outputFormat, VkExtent2D extent, double timeSeconds) {
    SceneFrame f{graph,  frame,  scene,      camera,      settings,         output,
                 outputFormat, extent, timeSeconds, ctx_.gpu(), scene.registry(), scene.environment()};
    prepareFrame(f);        // SceneRendererLights.cpp
    gatherObjects(f);       // SceneRendererObjects.cpp
    selectVirtualPages(f);  // SceneRendererObjects.cpp

    // ---- Render graph ----------------------------------------------------------------------
    const VkExtent2D render = f.renderExtent;  // up to TAA; the output's size after it
    f.gpuOcclusionCulling = settings.gpuOcclusionCulling && render.width > 0 && render.height > 0;
    f.sdfGlobalIllumination = settings.sdfGlobalIllumination && render.width > 0 && render.height > 0;
    f.restirDirectLighting = settings.restirDirectLighting && render.width > 0 && render.height > 0 &&
                             opaqueReflections_ && opaqueReflectionsTwoSided_ && opaqueReflectionsSkinned_;
    f.hdr = graph.createImage("SceneColorHDR", {ctx_.hdrFormat(), render.width, render.height});
    f.depth = graph.createImage("SceneDepth", {ctx_.depthFormat(), render.width, render.height});
    f.stats.renderWidth = render.width;
    f.stats.renderHeight = render.height;
    f.stateFor = {frame.bindingSet(ctx_.gpu(), ctx_.frameLayout(), ctx_.samplers()), ctx_.bindless().table()};
    addShadowPasses(f);    // SceneRendererShadows.cpp
    addDepthPasses(f);     // SceneRendererSurfaces.cpp
    addSurfacePasses(f);   // SceneRendererSurfaces.cpp
    addParticles(f);       // SceneRendererPost.cpp
    addPostProcessing(f);  // SceneRendererPost.cpp
    addOverlays(f);        // SceneRendererOverlays.cpp

    // Last frame's state for the next frame's motion vectors.
    previousViewProjection_ = f.unjittered;
    previousJitter_ = f.jitter;
    previousCameraPosition_ = camera.position;
    previousCameraForward_ = f.cameraForward;
    previousWorlds_ = std::move(f.currentWorlds);
    ++taaFrame_;
    return f.stats;
}

} // namespace lumora::render
