// Objects: gathering, culling, materials and draw lists; the virtual shadow map pages they need (see SceneRendererFrame.hpp).
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
#include "renderer/GpuTypes.hpp"
#include "renderer/LightGrid.hpp"
#include "renderer/LightProbes.hpp"
#include "renderer/RenderContext.hpp"
#include "scene/Landscape.hpp"
#include "scene/Scene.hpp"

#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>
#include <span>
#include <string>


namespace lumora::render {

namespace {

/// Joint matrices of a skinned mesh entity: inverse(mesh world) * joint world * inverse bind, for
/// every joint of its skin, appended to `out`. The joints are the entities that mirror the skin's
/// joint nodes under the same model root. False (palette of identities) if anything is missing.
bool appendSkinPalette(const Scene& scene, entt::entity entity, const MeshRendererComponent& mesh, AssetLibrary* assets,
                       std::vector<glm::mat4>& out) {
    const auto& reg = scene.registry();
    const auto* skinComponent = reg.try_get<SkinComponent>(entity);
    const ImportedModel* model = nullptr;
    if (skinComponent != nullptr && assets != nullptr) {
        if (auto ref = assets->resolveMeshKey(mesh.meshAsset)) {
            if (auto loaded = assets->model(ref->file)) model = *loaded;
        }
    }
    if (model == nullptr || skinComponent->skin < 0 || static_cast<std::size_t>(skinComponent->skin) >= model->skins.size()) {
        out.push_back(glm::mat4(1.0f));
        return false;
    }
    const ImportedSkin& skin = model->skins[static_cast<std::size_t>(skinComponent->skin)];
    // The model root: the first ancestor that does not mirror a model node (the mesh entity is a
    // node itself, or a material part under one).
    entt::entity root = scene.parentOf(entity);
    while (root != entt::null && reg.all_of<ModelNodeComponent>(root)) root = scene.parentOf(root);
    std::vector<entt::entity> nodes(model->nodes.size(), entt::entity{entt::null});
    auto collect = [&](auto& self, entt::entity e) -> void {
        for (entt::entity child : scene.childrenOf(e)) {
            if (const auto* node = reg.try_get<ModelNodeComponent>(child)) {
                if (node->node >= 0 && static_cast<std::size_t>(node->node) < nodes.size()) {
                    nodes[static_cast<std::size_t>(node->node)] = child;
                }
            }
            self(self, child);
        }
    };
    if (root != entt::null) collect(collect, root);
    const glm::mat4 toMesh = glm::inverse(scene.worldMatrix(entity));
    for (std::size_t j = 0; j < skin.joints.size(); ++j) {
        const int node = skin.joints[j];
        const entt::entity joint = node >= 0 && static_cast<std::size_t>(node) < nodes.size() ? nodes[static_cast<std::size_t>(node)] : entt::null;
        out.push_back(joint != entt::null ? toMesh * scene.worldMatrix(joint) * skin.inverseBind[j] : glm::mat4(1.0f));
    }
    return true;
}

} // namespace

void SceneRenderer::gatherObjects(SceneFrame& f) {
    using Draw = SceneDraw;
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const auto& reg = f.reg;
    const Scene& scene = f.scene;
    const CameraView& camera = f.camera;
    const RenderSettings& settings = f.settings;
    const VkExtent2D extent = f.extent;
    FrameResources& frame = f.frame;
    const FrameConstants& fc = f.fc;
    const bool anyShadows = f.anyShadows;
    auto& currentWorlds = f.currentWorlds;

    // ---- Objects + culling -----------------------------------------------------------------
    std::vector<Draw> draws;    // visible to the camera
    std::vector<Draw> casters;  // may cast into a shadow cascade (also off-screen objects)
    std::vector<Draw> transparents;  // see-through, visible to the camera (drawn after water)
    std::vector<GpuObject> objects;
    const Frustum frustum = Frustum::fromViewProjection(fc.viewProjection);

    // 1. Gather (serial: the ECS view and asset lookups, which may start loads, are not thread-safe).
    struct Candidate {
        entt::entity entity;
        const MeshRendererComponent* mesh;
        const glm::mat4* world;
        const GpuMesh* gpuMesh;
        std::shared_ptr<const MaterialAsset> material;  // the .lmat it uses, if any
        float pattern = -1.0f;  // overrides the surface pattern (3 = terrain)
        float extra = 0.0f;     // flags.z for the pattern (terrain: packed rock colour)
    };
    std::vector<Candidate> candidates;
    ++frameCounter_;
    releaseGraphPrograms(false);
    ctx_.textures().beginFrame();
    const std::filesystem::path projectRoot = settings.assets != nullptr ? settings.assets->root() : std::filesystem::path();
    auto landscapeEntry = [&](entt::entity e, std::uint64_t kind) -> LandscapeEntry& {
        const std::uint64_t slot = (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&scene)) * 0x9E3779B97F4A7C15ull) ^
                                   (static_cast<std::uint64_t>(entt::to_integral(e)) << 2) ^ kind;
        LandscapeEntry& entry = landscape_[slot];
        entry.lastFrame = frameCounter_;
        return entry;
    };
    // Terrains: a procedural mesh each, grass/rock coloured by slope in the shader.
    for (auto [e, terrain, world] : reg.view<TerrainComponent, WorldTransformComponent>().each()) {
        const std::uint64_t key = terrainKey(terrain, projectRoot) ^ (static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(terrain.textureTiling)) << 17);
        const GpuMesh* gpuMesh = ctx_.meshes().procedural("terrain:" + scene.uuidOf(e).toString(), key, [&] {
            return buildTerrainMesh(*terrainData(terrain, projectRoot), terrain.textureTiling);
        });
        if (gpuMesh == nullptr) continue;
        LandscapeEntry& entry = landscapeEntry(e, 1);
        entry.look.baseColor = terrain.grassColor;
        entry.look.roughness = 0.92f;
        entry.look.baseColorTexture = terrain.texture;
        auto channel = [](float c) { return static_cast<std::uint32_t>(std::lround(std::sqrt(std::clamp(c, 0.0f, 1.0f)) * 255.0f)); };
        const std::uint32_t rock = channel(terrain.rockColor.r) | (channel(terrain.rockColor.g) << 8) | (channel(terrain.rockColor.b) << 16);
        candidates.push_back({e, &entry.look, &world.matrix, gpuMesh, nullptr, 3.0f, static_cast<float>(rock)});
    }
    // Foliage: one candidate per copy; consecutive objects of one mesh become instanced draws.
    for (auto [e, foliage] : reg.view<FoliageComponent>().each()) {
        LandscapeEntry& entry = landscapeEntry(e, 2);
        const std::uint64_t key = foliageKey(scene, e, projectRoot);
        if (entry.key != key || entry.worlds.empty() != (foliage.count == 0)) {
            entry.key = key;
            entry.worlds = foliageInstances(scene, e, projectRoot);
        }
        entry.look.mesh = foliage.mesh;
        entry.look.meshAsset = foliage.meshAsset;
        entry.look.material = foliage.material;
        entry.look.baseColor = foliage.color;
        entry.look.roughness = 0.8f;
        const GpuMesh* gpuMesh = meshFor(entry.look, settings.assets, settings.blockOnAssets);
        if (gpuMesh == nullptr) continue;
        std::shared_ptr<const MaterialAsset> material;
        if (!foliage.material.empty() && settings.assets != nullptr) material = settings.assets->material(foliage.material);
        for (const glm::mat4& m : entry.worlds) candidates.push_back({e, &entry.look, &m, gpuMesh, material});
    }
    std::erase_if(landscape_, [&](const auto& item) { return item.second.lastFrame + 300 < frameCounter_; });
    // Meshes with blend shapes at non-zero weights: deformed on the CPU into one buffer for the
    // frame, each drawn from its own range (no instancing, no coarser levels of detail).
    frame.morphedMeshes.clear();
    std::vector<Vertex> morphVertices;
    std::vector<Vertex> morphScratch;
    auto morphed = [&](const MeshRendererComponent& mesh, const GpuMesh& base, const std::vector<float>& weights) -> const GpuMesh* {
        if (settings.assets == nullptr || mesh.meshAsset.empty()) return nullptr;
        if (std::ranges::none_of(weights, [](float w) { return std::abs(w) > 1e-5f; })) return nullptr;
        const auto request = settings.assets->requestMesh(mesh.meshAsset);  // loaded already: the GPU mesh exists
        if (request.value == nullptr || request.value->data.morphs.empty()) return nullptr;
        const MeshData& data = request.value->data;
        GpuMesh& out = frame.morphedMeshes.emplace_back();
        out.indices = base.indices;
        out.indexCount = base.indexCount;
        out.skin = base.skin;
        out.bounds = applyMorphTargets(data, weights, morphScratch);
        out.vertexOffset = morphVertices.size() * sizeof(Vertex);
        morphVertices.insert(morphVertices.end(), morphScratch.begin(), morphScratch.end());
        return &out;
    };
    for (auto [e, mesh, world] : reg.view<MeshRendererComponent, WorldTransformComponent>().each()) {
        if (!mesh.visible) continue;
        const GpuMesh* gpuMeshPtr = meshFor(mesh, settings.assets, settings.blockOnAssets);
        if (gpuMeshPtr == nullptr) continue;  // still loading, or unresolved (already reported)
        if (const auto* morph = reg.try_get<MorphWeightsComponent>(e)) {
            if (const GpuMesh* deformed = morphed(mesh, *gpuMeshPtr, morph->weights)) gpuMeshPtr = deformed;
        }
        std::shared_ptr<const MaterialAsset> material;
        if (!mesh.material.empty() && settings.assets != nullptr) material = settings.assets->material(mesh.material);
        candidates.push_back({e, &mesh, &world.matrix, gpuMeshPtr, std::move(material)});
    }

    if (!morphVertices.empty()) {
        nvrhi::IBuffer* buffer = frame.morphVertices.write(gpu, std::as_bytes(std::span(morphVertices)), DynamicBuffer::Kind::Vertex,
                                                           sizeof(Vertex), "Morphed vertices");
        for (GpuMesh& m : frame.morphedMeshes) m.vertices = buffer;
        stats.morphedObjects = static_cast<std::uint32_t>(frame.morphedMeshes.size());
    }

    // 2. Bounds, culling and per-object constants on the job system's threads (pure math).
    struct Prepared {
        glm::vec3 center;
        glm::vec3 extent;
        bool visible;
        GpuObject object;
    };
    std::vector<Prepared> prepared(candidates.size());
    JobSystem::global().parallelFor(candidates.size(), 512, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            const Candidate& c = candidates[i];
            const glm::mat4& world = *c.world;
            const glm::vec3 localCenter = (c.gpuMesh->bounds.min + c.gpuMesh->bounds.max) * 0.5f;
            const glm::vec3 localExtent = (c.gpuMesh->bounds.max - c.gpuMesh->bounds.min) * 0.5f;
            Prepared& p = prepared[i];
            p.center = glm::vec3(world * glm::vec4(localCenter, 1.0f));
            const glm::mat3 m(world);
            p.extent = glm::abs(m[0]) * localExtent.x + glm::abs(m[1]) * localExtent.y + glm::abs(m[2]) * localExtent.z;
            // Skinned meshes are never culled: animated limbs leave the bind-pose bounds.
            p.visible = !settings.frustumCulling || c.gpuMesh->skinned() || frustum.intersectsAabb(p.center, p.extent);
            if (!p.visible && !anyShadows) continue;
            const MeshRendererComponent& mesh = *c.mesh;
            const MaterialAsset* mat = c.material.get();
            GpuObject& obj = p.object;
            obj.world = world;
            obj.normalMatrix = glm::mat4(glm::transpose(glm::inverse(glm::mat3(world))));
            if (mat != nullptr) {
                obj.baseColorMetallic = glm::vec4(mat->baseColor, mat->metallic);
                obj.emissiveRoughness = glm::vec4(mat->emissive * mat->emissiveStrength, mat->roughness);
            } else {
                obj.baseColorMetallic = glm::vec4(mesh.baseColor, mesh.metallic);
                obj.emissiveRoughness = glm::vec4(mesh.emissive * mesh.emissiveStrength, mesh.roughness);
            }
            const float pattern = c.pattern >= 0.0f ? c.pattern
                                                    : static_cast<float>(mat != nullptr ? mat->pattern : static_cast<std::int32_t>(mesh.pattern));
            const bool highlighted = c.entity == settings.selected ||
                                     std::ranges::find(settings.alsoSelected, c.entity) != settings.alsoSelected.end();
            obj.flags = glm::vec4(highlighted ? 1.0f : 0.0f, pattern, c.extra, 0.0f);
        }
    });

    // 3. Textures (serial: the cache may start uploads) and the draw lists, in scene order.
    objects.reserve(candidates.size());
    std::vector<glm::mat4> joints;  // skinning palettes of every skinned object, back to back
    bool transmissiveDraws = false;  // see-through surfaces that need the scene behind them
    struct LodBlend {
        Draw draw;         // the finer level (its object index is set once it is appended)
        GpuObject object;  // a copy of the object that keeps the other half of the pixels
    };
    std::vector<LodBlend> lodBlends;
    // Objects grouped by mesh (unskinned first), so copies of a mesh sit next to each other.
    std::vector<std::size_t> order(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::ranges::stable_sort(order, std::less<>{}, [&](std::size_t i) {
        return std::pair(candidates[i].gpuMesh->skinned(), candidates[i].gpuMesh);
    });
    for (const std::size_t i : order) {
        Prepared& p = prepared[i];
        if (!p.visible) {
            ++stats.culledObjects;
            if (!anyShadows) continue;
        }
        GpuObject& obj = p.object;
        if (p.visible && settings.assets != nullptr) {  // casters only need depth, not materials
            const MeshRendererComponent& mesh = *candidates[i].mesh;
            const MaterialAsset* m = candidates[i].material.get();
            TextureCache& textures = ctx_.textures();
            const bool block = settings.blockOnAssets;
            const std::string& baseMap = m != nullptr ? m->baseColorTexture : mesh.baseColorTexture;
            const std::string& mrMap = m != nullptr ? m->metallicRoughnessTexture : mesh.metallicRoughnessTexture;
            const std::string& emissiveMap = m != nullptr ? m->emissiveTexture : mesh.emissiveTexture;
            const std::string& normalMap = m != nullptr ? m->normalTexture : mesh.normalTexture;
            // Texels across the object needs: its size on screen, with room for tiling.
            const float wanted = candidates[i].pattern >= 0.0f
                                     ? TextureCache::kFullResolution
                                     : 2.0f * static_cast<float>(extent.height) *
                                           screenCoverage(glm::length(p.extent), glm::length(p.center - camera.position), camera.fovYRadians);
            obj.textures = glm::uvec4(textures.index(*settings.assets, baseMap, TextureKind::Color, block, wanted),
                                      textures.index(*settings.assets, mrMap, TextureKind::Data, block, wanted),
                                      textures.index(*settings.assets, emissiveMap, TextureKind::Color, block, wanted),
                                      textures.index(*settings.assets, normalMap, TextureKind::Normal, block, wanted));
        } else if (settings.assets != nullptr) {
            // Off screen, only its shadow: the base colour's alpha still cuts out leaves (a small
            // copy is enough; without it a tree out of view would cast solid cards).
            const MaterialAsset* m = candidates[i].material.get();
            const std::string& baseMap = m != nullptr ? m->baseColorTexture : candidates[i].mesh->baseColorTexture;
            constexpr float kShadowTexels = 256.0f;
            obj.textures.x = ctx_.textures().index(*settings.assets, baseMap, TextureKind::Color, settings.blockOnAssets, kShadowTexels);
        }
        {
            // Last frame's transform of this object (by entity, mesh and copy) for motion vectors.
            std::uint64_t key = (static_cast<std::uint64_t>(entt::to_integral(candidates[i].entity)) << 32) ^
                                std::hash<const void*>{}(candidates[i].gpuMesh);
            while (currentWorlds.contains(key)) key = key * 1099511628211ull + 1;  // copies of one mesh
            const auto previous = previousWorlds_.find(key);
            obj.previousWorld = previous != previousWorlds_.end() && taaHistoryValid_ ? previous->second : obj.world;
            currentWorlds.emplace(key, obj.world);
        }
        if (candidates[i].gpuMesh->skinned()) {
            obj.flags.z = static_cast<float>(joints.size());
            obj.flags.w = appendSkinPalette(scene, candidates[i].entity, *candidates[i].mesh, settings.assets, joints) ? 1.0f : 0.0f;
        }
        // Level of detail: the coarsest whose error stays under a pixel on screen (shadows use
        // the same). Errors are relative to the mesh's largest side: scaled like the object.
        const GpuMesh* drawn = candidates[i].gpuMesh;
        LodChoice lod;
        if (!drawn->lods.empty() && settings.lodBias > 0.0f) {
            const float largestSide = 2.0f * std::max({p.extent.x, p.extent.y, p.extent.z});
            std::array<float, 8> errors{};
            const std::size_t levels = std::min(drawn->lods.size(), errors.size());
            for (std::size_t l = 0; l < levels; ++l) errors[l] = drawn->lods[l].lodError * largestSide;
            const float pixelsPerUnit = static_cast<float>(extent.height) / (2.0f * std::tan(camera.fovYRadians * 0.5f));
            lod = selectLod(std::span(errors.data(), levels), glm::length(p.center - camera.position) - glm::length(p.extent),
                            pixelsPerUnit, settings.lodBias);
            drawn = &drawn->level(lod.level);
            if (lod.level > 0 && p.visible) ++stats.lodObjects;
        }
        std::uint64_t state = std::hash<const void*>{}(drawn);
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                state = state * 1099511628211ull ^ std::bit_cast<std::uint32_t>(obj.world[col][row]);
            }
        }
        const MeshRendererComponent& look = *candidates[i].mesh;
        const MaterialAsset* lmat = candidates[i].material.get();
        float cutoff = 0.0f;
        bool twoSided = false;
        float opacity = 1.0f;
        bool transparent = false;
        bool transmissive = false;
        std::optional<ModelSurface> surface;
        if (lmat != nullptr) {
            // A material asset replaces everything: the renderer's settings and the model's material.
            surface = materialSurface(*lmat);
            cutoff = surface->alphaCutoff;
            twoSided = surface->twoSided;
            opacity = surface->opacity;
            transparent = surface->transparent;
        } else {
            cutoff = look.alphaCutoff;
            twoSided = look.doubleSided;
            opacity = std::clamp(look.opacity, 0.0f, 1.0f);
            transparent = opacity < 1.0f;
            if (!look.meshAsset.empty() && settings.assets != nullptr) {
                surface = modelSurface(*settings.assets, look.meshAsset);
                if (cutoff < 0.0f) cutoff = surface->alphaCutoff;
                twoSided = twoSided || surface->twoSided;
                transparent = transparent || surface->transparent;
                opacity *= surface->opacity;
            }
        }
        if (surface) {
            // Volume lengths are in the object's units: scale them like the object.
            const float scale = std::cbrt(std::abs(glm::determinant(glm::mat3(obj.world))));
            obj.uvTransformU = surface->uvTransformU;
            obj.uvTransformV = surface->uvTransformV;
            obj.specularClearcoat = surface->specularClearcoat;
            obj.sheen = surface->sheen;
            obj.transmission = surface->transmission * glm::vec4(1.0f, 1.0f, scale, scale);
            obj.attenuation = surface->attenuation;
            transmissive = surface->transmissive;
            transparent = transparent || transmissive;
        }
        if (lmat != nullptr && lmat->program) {
            // A material graph: its programs, and the bindless indices of its textures.
            const GraphProgramGpu& program = graphProgram(lmat->program);
            obj.graph = glm::uvec4(program.fragmentIndex, lmat->program->fragmentInstructions(),
                                   program.vertexIndex == UINT32_MAX ? kNoTexture : program.vertexIndex,
                                   lmat->program->vertexInstructions());
            for (std::size_t t = 0; t < lmat->program->textures.size() && t < matgraph::kMaxTextures; ++t) {
                const MaterialProgram::Texture& texture = lmat->program->textures[t];
                const TextureKind kind = texture.kind == matgraph::TextureKind::Normal ? TextureKind::Normal
                                         : texture.kind == matgraph::TextureKind::Data ? TextureKind::Data
                                                                                       : TextureKind::Color;
                obj.graphTextures[t / 4][static_cast<int>(t % 4)] =
                    settings.assets != nullptr ? ctx_.textures().index(*settings.assets, texture.key, kind, settings.blockOnAssets) : kNoTexture;
            }
            // Moving vertices (wind) change the shadows every frame: never reuse cached tiles.
            if (obj.graph.z != kNoTexture) state ^= (frameCounter_ + 1) * 0x9E3779B97F4A7C15ull;
        }
        const bool masked = !transparent && cutoff > 0.0f && (obj.textures.x != kNoTexture || obj.graph.x != kNoTexture);
        std::uint32_t occlusion = kNoTexture;
        const std::string& occlusionMap = lmat != nullptr ? lmat->occlusionTexture : look.occlusionTexture;
        if (p.visible && settings.assets != nullptr && !occlusionMap.empty()) {
            occlusion = ctx_.textures().index(*settings.assets, occlusionMap, TextureKind::Data, settings.blockOnAssets);
        }
        // x: cutoff (opaque) or opacity (see-through); y: bit 0 two-sided, bit 1 see-through, bit 2 transmissive.
        obj.surface = glm::vec4(transparent ? opacity : masked ? cutoff : 0.0f,
                                static_cast<float>((twoSided ? 1 : 0) | (transparent ? 2 : 0) | (transmissive ? 4 : 0)),
                                std::bit_cast<float>(occlusion), lmat != nullptr ? lmat->normalStrength : look.normalStrength);
        const auto variant = static_cast<std::uint8_t>((masked ? kMasked : 0) | (twoSided ? kTwoSided : 0));
        // Cached shadows (virtual pages, local tiles) are redrawn when this changes: a caster
        // drawn solid while its map was loading must be drawn again cut out once it is there.
        const bool mapped = obj.textures.x != kNoTexture || obj.graph.x != kNoTexture;
        state = (state ^ (variant | (mapped ? 0x10u : 0u) | (transparent ? 0x20u : 0u))) * 1099511628211ull;
        state = (state ^ std::bit_cast<std::uint32_t>(cutoff)) * 1099511628211ull;
        const Draw draw{drawn,
                        static_cast<std::uint32_t>(objects.size()),
                        p.center,
                        p.extent,
                        1,
                        static_cast<std::uint64_t>(entt::to_integral(candidates[i].entity)),
                        state,
                        candidates[i].gpuMesh->skinned() ? std::uint8_t{0} : variant};
        if (transparent) {
            if (p.visible && opacity > 0.0f) {
                transparents.push_back(draw);
                transmissiveDraws = transmissiveDraws || transmissive;
            }
            // Blended leaves and fences: a cut-out shadow where the map (or graph) makes them at
            // least half opaque. Glass (transmission, or one opacity without a map) casts none.
            if (anyShadows && mapped && !transmissive && opacity > 0.0f && !candidates[i].gpuMesh->skinned()) {
                Draw caster = draw;
                caster.variant = static_cast<std::uint8_t>(kMasked | (twoSided ? kTwoSided : 0));
                casters.push_back(caster);
            }
        } else {
            // LOD blending (opaque objects in view): the finer level is drawn too, and the two
            // split the pixels by a dither pattern (lodDitheredOut in the shaders). Shadows keep
            // the coarser level alone: cached shadow tiles would not follow a screen pattern.
            const bool blend = p.visible && settings.lodBlending && lod.level > 0 && lod.fade > 0.0f;
            if (blend) {
                const bool skinned = candidates[i].gpuMesh->skinned();
                Draw coarse = draw;
                // The masked depth prepass can discard; with no cutoff it cuts nothing else.
                if (!skinned) coarse.variant = static_cast<std::uint8_t>(coarse.variant | kMasked);
                Draw finer = coarse;
                finer.mesh = &candidates[i].gpuMesh->level(lod.level - 1);
                GpuObject finerObject = obj;
                obj.lod = glm::vec4(lod.fade, 0.0f, 0.0f, 0.0f);
                finerObject.lod = glm::vec4(lod.fade, 1.0f, 0.0f, 0.0f);
                draws.push_back(coarse);
                lodBlends.push_back({finer, finerObject});
                ++stats.lodBlendingObjects;
            } else if (p.visible) {
                draws.push_back(draw);
            }
            if (anyShadows) casters.push_back(draw);
        }
        objects.push_back(obj);
    }
    // The finer levels of blending objects: their own objects, consecutive per mesh, so copies
    // of one mesh (foliage) still become one instanced draw.
    std::ranges::stable_sort(lodBlends, std::less<>{}, [](const LodBlend& b) { return b.draw.mesh; });
    for (LodBlend& b : lodBlends) {
        b.draw.object = static_cast<std::uint32_t>(objects.size());
        objects.push_back(b.object);
        draws.push_back(b.draw);
    }
    // See-through surfaces back to front (by bounds centre), so nearer ones blend over farther.
    std::ranges::stable_sort(transparents, std::greater<>{}, [&](const Draw& d) { return glm::length(d.center - camera.position); });
    // Unskinned first, then skinned (one pipeline switch); within each, grouped by mesh.
    auto byPipelineAndMesh = [](const Draw& d) { return std::tuple(d.mesh->skinned(), d.variant, d.mesh); };
    std::ranges::stable_sort(draws, std::less<>{}, byPipelineAndMesh);
    std::ranges::stable_sort(casters, std::less<>{}, byPipelineAndMesh);
    // Decals are objects too: world, inverse world (in normalMatrix), colour + opacity, texture.
    std::vector<std::uint32_t> decals;
    for (auto [e, decal, world] : reg.view<DecalComponent, WorldTransformComponent>().each()) {
        const glm::vec3 center(world.matrix[3]);
        const glm::mat3 m(world.matrix);
        const glm::vec3 halfExtent = (glm::abs(m[0]) + glm::abs(m[1]) + glm::abs(m[2])) * 0.5f;
        if (settings.frustumCulling && !frustum.intersectsAabb(center, halfExtent)) continue;
        GpuObject obj;
        obj.world = world.matrix;
        obj.normalMatrix = glm::inverse(world.matrix);
        obj.baseColorMetallic = glm::vec4(decal.color, std::clamp(decal.opacity, 0.0f, 1.0f));
        if (!decal.texture.empty() && settings.assets != nullptr) {
            obj.textures.x = ctx_.textures().index(*settings.assets, decal.texture, TextureKind::Color, settings.blockOnAssets);
        }
        decals.push_back(static_cast<std::uint32_t>(objects.size()));
        objects.push_back(obj);
    }
    // Water surfaces: objects too (a unit grid scaled to the surface; wave settings in flags).
    std::vector<std::uint32_t> waters;
    const GpuMesh* waterGrid = nullptr;
    for (auto [e, water, world] : reg.view<WaterComponent, WorldTransformComponent>().each()) {
        const glm::mat4 surface = world.matrix * glm::scale(glm::mat4(1.0f), glm::vec3(water.width, 1.0f, water.length));
        const glm::vec3 center(surface[3]);
        const glm::mat3 m(surface);
        const glm::vec3 halfExtent = (glm::abs(m[0]) + glm::abs(m[2])) * 0.5f + glm::vec3(0.0f, water.waveHeight * 1.5f, 0.0f);
        if (settings.frustumCulling && !frustum.intersectsAabb(center, halfExtent)) continue;
        if (waterGrid == nullptr) waterGrid = ctx_.meshes().procedural("water:grid", 256, [] { return buildWaterGrid(256); });
        GpuObject obj;
        obj.world = surface;
        obj.baseColorMetallic = glm::vec4(water.shallowColor, 0.0f);
        obj.emissiveRoughness = glm::vec4(water.deepColor, 0.0f);
        obj.flags = glm::vec4(water.waveHeight, water.waveLength, water.waveSpeed, water.clarity);
        waters.push_back(static_cast<std::uint32_t>(objects.size()));
        objects.push_back(obj);
    }
    if (objects.empty()) objects.emplace_back();  // structured buffers may not be empty
    if (joints.empty()) joints.emplace_back(1.0f);
    frame.objects.write(gpu, std::as_bytes(std::span(objects)), DynamicBuffer::Kind::Structured, sizeof(GpuObject),
                        "FrameObjects");
    frame.joints.write(gpu, std::as_bytes(std::span(joints)), DynamicBuffer::Kind::Structured, sizeof(glm::mat4),
                       "FrameJoints");
    stats.visibleObjects = static_cast<std::uint32_t>(draws.size() - lodBlends.size());  // a blending object once
    for (const Draw& d : draws) {
        stats.triangles += d.mesh->indexCount / 3;
    }
    draws = instanced(draws);
    stats.drawCalls = static_cast<std::uint32_t>(draws.size()) * 2 + 2 + (settings.showGrid ? 1u : 0u);  // prepass + shading


    f.draws = std::move(draws);
    f.casters = std::move(casters);
    f.transparents = std::move(transparents);
    f.transmissiveDraws = transmissiveDraws;
    f.decals = std::move(decals);
    f.waters = std::move(waters);
    f.waterGrid = waterGrid;
}

void SceneRenderer::selectVirtualPages(SceneFrame& f) {
    using Draw = SceneDraw;
    nvrhi::IDevice* gpu = f.gpu;
    SceneDrawStats& stats = f.stats;
    const CameraView& camera = f.camera;
    const RenderSettings& settings = f.settings;
    FrameResources& frame = f.frame;
    FrameConstants& fc = f.fc;
    const bool virtualShadows = f.virtualShadows;
    const std::vector<Draw>& casters = f.casters;
    const glm::vec3 sunTravel = f.sunTravel;

    // ---- Virtual shadow map: pages the view needs, what changed, the page table --------------
    VsmFrame vsmFrame;
    // This frame's slot of GPU page requests: read what it holds (written two frames ago, if
    // that frame marked any), then it is free for this frame's marks.
    VsmRequests& previous = vsmRequests_[vsmRequestFrame_ % vsmRequests_.size()];
    std::optional<VsmFeedback> feedback;
    if (virtualShadows) {
        std::vector<VsmCaster> vsmCasters;
        vsmCasters.reserve(casters.size());
        for (const Draw& d : casters) vsmCasters.push_back({d.key, d.center, d.extent, d.state, d.mesh->skinned()});
        const std::uint32_t budget = settings.blockOnAssets ? 0u : settings.virtualShadowPageBudget;
        // Requests the visible surfaces made two frames ago (that frame has finished).
        // Only for (nearly) the same view: another camera's requests (a thumbnail, a bake) would
        // leave this view's pages out. Single captures always take the whole frustum.
        auto sameView = [&](const glm::mat4& a, const glm::mat4& b) {
            for (int c = 0; c < 4; ++c) {
                if (glm::length(a[c] - b[c]) > 0.02f * (glm::length(b[c]) + 1.0f)) return false;
            }
            return true;
        };
        if (previous.pending && settings.virtualShadowGpuRequests && !settings.blockOnAssets &&
            sameView(previous.viewProjection, fc.viewProjection)) {
            std::size_t pitch = 0;
            const auto* mapped = static_cast<const std::uint8_t*>(
                gpu->mapStagingTexture(previous.staging, nvrhi::TextureSlice(), nvrhi::CpuAccessMode::Read, &pitch));
            if (mapped != nullptr) {
                const std::size_t rows = static_cast<std::size_t>(VirtualShadowMap::kPagesPerLevel) * static_cast<std::size_t>(previous.levels);
                feedback = VirtualShadowMap::decodeRequests(
                    std::span(reinterpret_cast<const std::uint32_t*>(mapped), (rows - 1) * (pitch / 4) + VirtualShadowMap::kPagesPerLevel),
                    pitch / 4, previous.levels, previous.windowMin, previous.generation);
                gpu->unmapStagingTexture(previous.staging);
            }
        }
        vsmFrame = vsm_.update(camera, sunTravel, settings.shadowDistance, vsmCasters, budget, feedback ? &*feedback : nullptr);
        fc.vsmLightView = vsmFrame.lightView;
        fc.vsmParams = glm::vec4(static_cast<float>(vsmFrame.levels), vsmFrame.firstHalf, vsmFrame.levelReach,
                                 static_cast<float>(VirtualShadowMap::kPagesPerLevel));
        fc.vsmDepth = glm::vec4(vsmFrame.depthMin, vsmFrame.depthScale, static_cast<float>(VirtualShadowMap::kPageTexels),
                                static_cast<float>(VirtualShadowMap::kPoolPages));
        stats.virtualPagesRequested = vsmFrame.requested;
        stats.virtualPagesCached = vsmFrame.cached;
        stats.virtualPagesRendered = static_cast<std::uint32_t>(vsmFrame.render.size());
    }
    previous.pending = false;  // consumed, or stale
    frame.constants.write(gpu, std::as_bytes(std::span(&fc, 1)), DynamicBuffer::Kind::Constant, sizeof(FrameConstants),
                          "FrameConstants");
    f.vsmFrame = std::move(vsmFrame);
}

} // namespace lumora::render
