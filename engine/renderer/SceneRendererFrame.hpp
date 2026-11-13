// Internal to the scene renderer: the state one SceneRenderer::build() call hands from phase to
// phase, and helpers the phases share. Only SceneRenderer*.cpp include it.
//
//   SceneRenderer.cpp          build() (the order of the phases), pipelines, caches, shadow math
//   SceneRendererLights.cpp    frame constants: camera, TAA jitter, sky, probes, lights, shadow setup
//   SceneRendererObjects.cpp   objects: gathering, culling, materials, draw lists; virtual shadow pages
//   SceneRendererShadows.cpp   passes: sun cascades, virtual shadow pages, local shadow atlas
//   SceneRendererSurfaces.cpp  passes: depth prepass, AO, contact shadows, fog, sky, opaque, decals,
//                              SSR, water, see-through surfaces, grid
//   SceneRendererPost.cpp      passes: particles, sun shafts, TAA, depth of field, motion blur,
//                              exposure, bloom, tone mapping, FXAA
//   SceneRendererOverlays.cpp  passes: gizmo overlay and debug shapes, sprites, HUD
#pragma once

#include "renderer/FrameResources.hpp"
#include "renderer/GpuTypes.hpp"
#include "renderer/MeshCache.hpp"
#include "renderer/RenderGraph.hpp"
#include "renderer/SceneRenderer.hpp"
#include "renderer/VirtualShadowMap.hpp"
#include "scene/Scene.hpp"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace lumora::render {

inline constexpr nvrhi::Format kAoFormat = nvrhi::Format::R8_UNORM;
inline constexpr nvrhi::Format kMotionFormat = nvrhi::Format::RG16_FLOAT;
// Per-pixel surface data for screen-space reflections (normal + roughness, specular weight).
inline constexpr nvrhi::Format kSurfaceFormat = nvrhi::Format::RGBA16_FLOAT;

/// One draw of a mesh, instanced over consecutive objects.
struct SceneDraw {
    const GpuMesh* mesh;
    std::uint32_t object;
    glm::vec3 center;  // world AABB
    glm::vec3 extent;
    std::uint32_t count = 1;  // instances: objects object .. object + count - 1
    std::uint64_t key = 0;    // the entity (virtual shadow map caster tracking)
    std::uint64_t state = 0;  // changes when its shadow can change
    std::uint8_t variant = 0; // kMasked | kTwoSided (unskinned meshes)
};
inline constexpr std::uint8_t kMasked = 1;
inline constexpr std::uint8_t kTwoSided = 2;

/// Copies of one mesh in consecutive objects become one instanced draw.
inline std::vector<SceneDraw> instanced(const std::vector<SceneDraw>& list) {
    std::vector<SceneDraw> out;
    out.reserve(list.size());
    for (const SceneDraw& d : list) {
        if (!out.empty() && out.back().mesh == d.mesh && out.back().variant == d.variant && out.back().object + out.back().count == d.object) {
            ++out.back().count;
        } else {
            out.push_back(d);
        }
    }
    return out;
}

/// A tile of the local (spot / point light) shadow atlas this frame.
struct LocalShadowTile {
    glm::mat4 viewProjection;
    glm::vec3 lightPosition;
    float range;
    std::uint32_t index;  // in the atlas
};

/// Every draw: a pipeline over the pass's framebuffer with set 0 and the bindless set 1.
struct DrawState {
    nvrhi::IBindingSet* frameSet = nullptr;
    nvrhi::IDescriptorTable* bindless = nullptr;
    nvrhi::GraphicsState operator()(nvrhi::IGraphicsPipeline* pipeline, const PassContext& ctx) const {
        nvrhi::GraphicsState state;
        state.pipeline = pipeline;
        state.framebuffer = ctx.framebuffer;
        state.viewport = ctx.viewport();
        state.bindings = {frameSet, bindless};
        return state;
    }
};

/// A mesh's vertex (and, when skinned, skin) stream and its index buffer.
inline void bindMesh(nvrhi::GraphicsState& state, const GpuMesh& mesh) {
    state.vertexBuffers.resize(0);
    state.vertexBuffers.push_back(nvrhi::VertexBufferBinding().setBuffer(mesh.vertices).setSlot(0).setOffset(mesh.vertexOffset));
    if (mesh.skinned()) state.vertexBuffers.push_back(nvrhi::VertexBufferBinding().setBuffer(mesh.skin).setSlot(1));
    state.indexBuffer = nvrhi::IndexBufferBinding().setBuffer(mesh.indices).setFormat(nvrhi::Format::R32_UINT);
}

inline void push(nvrhi::ICommandList* cmd, const PushConstants& pc) { cmd->setPushConstants(&pc, sizeof(pc)); }

/// A full-screen triangle with one pipeline and its push constants.
inline void fullscreen(nvrhi::ICommandList* cmd, nvrhi::GraphicsState state, const PushConstants& pc) {
    cmd->setGraphicsState(state);
    push(cmd, pc);
    cmd->draw(nvrhi::DrawArguments().setVertexCount(3));
}

/// What one build() call knows, filled phase by phase (see the file list above).
struct SceneFrame {
    // ---- Inputs
    RenderGraph& graph;
    FrameResources& frame;
    const Scene& scene;
    const CameraView& camera;
    const RenderSettings& settings;
    RGImage output;
    nvrhi::Format outputFormat;
    VkExtent2D extent;
    double timeSeconds;
    nvrhi::IDevice* gpu;
    const entt::registry& reg;
    const EnvironmentSettings& env;
    SceneDrawStats stats{};

    // ---- prepareFrame
    FrameConstants fc{};
    bool temporal = false;              // TAA this frame (jittered projection)
    glm::mat4 unjittered{1.0f};         // the view-projection without jitter
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
    glm::vec2 jitter{0.0f};
    std::unordered_map<std::uint64_t, glm::mat4> currentWorlds;  // becomes previousWorlds_ at the end
    bool volumetricFog = false;
    glm::uvec3 froxels{0u};
    bool haveSun = false;
    glm::vec3 sunTravel{0.0f, -1.0f, 0.0f};
    bool shadows = false;         // sun shadows
    bool virtualShadows = false;  // ... from the virtual shadow map
    bool localShadows = false;    // spot / point light shadows
    bool anyShadows = false;
    bool reflections = false;     // screen-space reflections
    std::vector<LocalShadowTile> localTiles;
    std::uint32_t localAtlasSize = 0;
    std::uint32_t tile = 0;       // a sun cascade's side in the atlas
    std::uint32_t atlasSize = 0;
    std::array<glm::mat4, kShadowCascades> cascadeMatrices{};

    // ---- gatherObjects, selectVirtualPages
    std::vector<SceneDraw> draws;         // opaque, visible to the camera
    std::vector<SceneDraw> casters;       // may cast into a shadow (also off-screen objects)
    std::vector<SceneDraw> transparents;  // see-through, visible, back to front
    bool transmissiveDraws = false;
    std::vector<std::uint32_t> decals;
    std::vector<std::uint32_t> waters;
    const GpuMesh* waterGrid = nullptr;
    VsmFrame vsmFrame;

    // ---- The graph's images
    RGImage hdr{};    // the scene's colour (replaced by SSR's output when it runs)
    RGImage depth{};
    DrawState stateFor;
    RGImage shadowMap{}, vsmPool{}, vsmPageTable{}, localAtlas{};
    RGImage motion{}, ao{}, contact{}, fogVolume{};
    bool ambientOcclusion = false;
};

} // namespace lumora::render
