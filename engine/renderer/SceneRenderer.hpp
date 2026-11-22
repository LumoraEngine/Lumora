#pragma once

#include "renderer/HudRenderer.hpp"
#include "scene/DebugDraw.hpp"

#include "renderer/GpuTypes.hpp"
#include "renderer/Overlay.hpp"
#include "renderer/RenderGraph.hpp"
#include "renderer/Sprites.hpp"
#include "renderer/VirtualShadowMap.hpp"
#include "scene/Camera.hpp"
#include "scene/Components.hpp"

#include <entt/entity/entity.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace lumora {
class AssetLibrary;
class Scene;
struct MaterialAsset;
struct MaterialProgram;
}

namespace lumora::render {

class GpuOcclusionCuller;
class RenderContext;
class RestirPass;
class SdfGiPass;
struct FrameResources;

struct RenderSettings {
    bool showGrid = false;
    float gridCellSize = 1.0f;
    float gridFadeDistance = 150.0f;
    bool frustumCulling = true;
    bool shadows = true;
    bool bloom = true;
    float shadowDistance = 50.0f;          // metres covered by the sun shadow cascades
    std::uint32_t shadowResolution = 2048; // per cascade; the atlas holds 2x2 cascades
    float cascadeSplitLambda = 0.8f;       // 0 = uniform cascade splits, 1 = logarithmic
    bool debugCascades = false;           // tint surfaces by the shadow cascade they sample
    /// Sun shadows from a virtual shadow map (clipmap pages cached across frames) instead of
    /// cascades, when the scene's environment asks for it too.
    bool virtualShadows = true;
    /// Virtual shadow map pages drawn per frame at most (the rest follow); 0 = all of them.
    std::uint32_t virtualShadowPageBudget = 256;
    /// The visible surfaces choose the virtual shadow map's fine pages (marked on the GPU, read
    /// back two frames later) instead of everything under the view frustum.
    bool virtualShadowGpuRequests = true;
    entt::entity selected = entt::null;
    std::vector<entt::entity> alsoSelected;  // the rest of a multi-selection (outlined too)
    const OverlayGeometry* overlay = nullptr;  // drawn on top after tone mapping (gizmos)
    const SpriteBatch* sprites = nullptr;      // pictures floating in the world (editor icons), on top
    const HudCanvas* hud = nullptr;            // the game's 2D overlay (text, rectangles), drawn last
    const DebugDraw* debug = nullptr;          // behaviours' debug shapes, drawn on top like the overlay
    AssetLibrary* assets = nullptr;            // resolves MeshRenderer::meshAsset; null = primitives only
    /// Wait for assets instead of skipping/untexturing them while they load (captures, headless).
    bool blockOnAssets = false;
    /// Passes with at least this many draws record on several threads (secondary command
    /// buffers); fewer are cheaper to record on one. 0 = always, UINT32_MAX = never.
    std::uint32_t parallelRecordingMinDraws = 256;
    /// Screen-space contact shadows: short rays towards the sun through the depth buffer, for the
    /// small shadows the shadow map is too coarse for.
    bool contactShadows = true;
    float contactShadowLength = 0.3f;  // metres at 10 m from the camera (scales with distance)
    /// Screen-space ambient occlusion: darkens ambient light where geometry meets.
    bool ambientOcclusion = true;
    float aoRadius = 0.6f;     // metres
    float aoIntensity = 1.6f;  // exponent applied to the visibility
    /// FXAA on the final image: smooths jagged edges (skipped while TAA runs).
    bool antiAliasing = true;
    /// Temporal anti-aliasing: frames are jittered and blended over time into a supersampled
    /// image. Needs consecutive frames of one view, so viewports turn it on; single captures
    /// (thumbnails, headless renders) leave it off.
    bool temporalAntiAliasing = false;
    /// With TAA: the scene renders at this fraction of the output's width and height and TAA
    /// upscales it (TAAU), filling in detail over the jittered frames. 0.5 .. 1; 1 = native.
    float renderScale = 1.0f;
    /// Screen-space reflections on glossy opaque surfaces (the scene on screen replaces the
    /// environment where the reflected ray finds it). Glossy rather than mirror-only with TAA.
    bool screenSpaceReflections = false;
    float reflectionDistance = 60.0f;  // metres a reflected ray travels at most
    float reflectionRoughness = 0.7f;  // roughness where screen-space reflections have faded out
    /// Reflection probes light glossy surfaces inside their boxes (off while baking them).
    bool reflectionProbes = true;
    /// Baked light probe volumes replace the sky's ambient light inside their boxes.
    bool lightProbeVolumes = true;
    /// The output gets the scene's linear HDR radiance instead of the tone-mapped image
    /// (no bloom blend, grading, vignette or dithering): probe baking reads it back.
    bool linearOutput = false;
    /// Levels of detail: imported meshes switch to simpler versions while the difference stays
    /// under a pixel on screen (under 1 / lodBias pixels). Larger values keep detail longer; 0
    /// always draws full detail. Near a switch both levels are drawn, splitting the pixels with
    /// a dither pattern, so the change is a cross-fade rather than a pop (lodBlending).
    float lodBias = 1.0f;
    bool lodBlending = true;
    /// The frame's intermediate images share textures when they are not needed at the same time
    /// (less video memory). False gives each its own (for comparing, or hunting a bad pass).
    bool shareTransientTextures = true;
    /// Two-phase Hi-Z GPU occlusion culling (builds a reversed-Z Hi-Z pyramid after the depth
    /// prepass and dispatches Phase-1 / Phase-2 indirect culling compute passes).
    bool gpuOcclusionCulling = false;
    /// SDF scene voxelization and ray-traced global illumination compute pass.
    bool sdfGlobalIllumination = false;
    /// Spatio-temporal ReSTIR direct lighting compute pass for point and spot lights.
    bool restirDirectLighting = false;
};

struct SceneDrawStats {
    std::uint32_t drawCalls = 0;
    std::uint32_t triangles = 0;
    std::uint32_t visibleObjects = 0;
    std::uint32_t culledObjects = 0;
    std::uint32_t transparentObjects = 0;  // see-through objects drawn (blended, back to front)
    std::uint32_t morphedObjects = 0;      // meshes deformed by their blend shapes this frame
    std::uint32_t lodObjects = 0;  // visible objects drawn with a simpler level of detail
    std::uint32_t lodBlendingObjects = 0;  // of those, ones cross-fading with the finer level (drawn twice)
    std::uint32_t lights = 0;
    std::uint32_t particles = 0;
    bool parallelRecording = false;  // a pass was set up to record on several threads
    bool shadowsRendered = false;
    bool virtualShadows = false;              // the sun used the virtual shadow map
    bool temporalAntiAliasing = false;        // TAA resolved the frame (with history when valid)
    std::uint32_t renderWidth = 0;            // the scene's images before TAA (smaller when it upscales)
    std::uint32_t renderHeight = 0;
    bool screenSpaceReflections = false;
    std::uint32_t reflectionProbes = 0;       // baked probes the frame used
    std::uint32_t lightProbeVolumes = 0;      // baked light probe volumes the frame used
    bool volumetricFog = false;               // froxel fog injected and integrated this frame
    bool autoExposure = false;                // exposure measured and adapted on the GPU
    bool motionBlur = false;
    bool contactShadows = false;
    bool virtualPagesFromSurfaces = false;    // the VSM's fine pages came from GPU requests
    std::uint32_t localShadowTiles = 0;       // spot / point light shadow tiles in use
    std::uint32_t localShadowTilesDrawn = 0;  // of those, drawn this frame (the rest cached)
    std::uint32_t virtualPagesRequested = 0;  // pages the view needed
    std::uint32_t virtualPagesCached = 0;     // of those, kept from earlier frames
    std::uint32_t virtualPagesRendered = 0;   // drawn this frame
    bool environmentLit = false;  // an HDR environment lit the frame
    bool ambientOcclusion = false;
    bool gpuOcclusionCulling = false;
    bool sdfGlobalIllumination = false;
    bool restirDirectLighting = false;
};

/// Orthographic sun transform covering the camera frustum up to `distance` (one cascade),
/// snapped to shadow-map texels so the shadows do not shimmer as the camera moves.
struct ShadowFit {
    glm::mat4 viewProjection{1.0f};
    float radius = 0.0f;
};
[[nodiscard]] ShadowFit fitSunShadow(const CameraView& camera, const glm::vec3& lightDirection, float distance,
                                     std::uint32_t resolution);
/// View distance at which each shadow cascade ends ("practical" split scheme: a blend of
/// logarithmic and uniform splits). The last split is `distance`.
[[nodiscard]] std::array<float, kShadowCascades> cascadeSplits(float nearPlane, float distance, float lambda);

/// Six-plane (five for infinite reversed-Z) frustum extracted from a view-projection matrix.
struct Frustum {
    glm::vec4 planes[5];
    [[nodiscard]] static Frustum fromViewProjection(const glm::mat4& viewProjection);
    [[nodiscard]] bool intersectsAabb(const glm::vec3& center, const glm::vec3& extents) const;
};

struct SceneFrame;

/// Translates a Scene into render-graph passes: shadows, depth prepass, sky, opaque (PBR forward),
/// see-through surfaces, post-processing, tone mapping and overlays. build() runs the phases in
/// order; each lives in its own file (see SceneRendererFrame.hpp).
class SceneRenderer {
public:
    explicit SceneRenderer(RenderContext& context);
    ~SceneRenderer();
    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    SceneDrawStats build(RenderGraph& graph, FrameResources& frame, const Scene& scene, const CameraView& camera,
                         const RenderSettings& settings, RGImage output, nvrhi::Format outputFormat, VkExtent2D extent,
                         double timeSeconds);

    [[nodiscard]] GpuOcclusionCuller* gpuCulling() noexcept;
    [[nodiscard]] const GpuOcclusionCuller* gpuCulling() const noexcept { return gpuCulling_.get(); }
    [[nodiscard]] SdfGiPass* sdfGi() noexcept;
    [[nodiscard]] const SdfGiPass* sdfGi() const noexcept { return sdfGi_.get(); }
    [[nodiscard]] RestirPass* restir() noexcept;
    [[nodiscard]] const RestirPass* restir() const noexcept { return restir_.get(); }
    [[nodiscard]] nvrhi::ITexture* sdfGiOutput() const noexcept { return sdfGiOutput_; }

private:
    // The phases of build(), in order.
    void prepareFrame(SceneFrame& f);        // SceneRendererLights.cpp
    void gatherObjects(SceneFrame& f);       // SceneRendererObjects.cpp
    void selectVirtualPages(SceneFrame& f);  // SceneRendererObjects.cpp
    void addShadowPasses(SceneFrame& f);     // SceneRendererShadows.cpp
    void addDepthPasses(SceneFrame& f);      // SceneRendererSurfaces.cpp
    void addSurfacePasses(SceneFrame& f);    // SceneRendererSurfaces.cpp
    void addParticles(SceneFrame& f);        // SceneRendererPost.cpp
    void addPostProcessing(SceneFrame& f);   // SceneRendererPost.cpp
    void addOverlays(SceneFrame& f);         // SceneRendererOverlays.cpp

    /// What an imported mesh's material says about cut-out and two-sidedness (cached per key).
    struct ModelSurface {
        float alphaCutoff = 0.0f;
        bool twoSided = false;
        bool transparent = false;  // glTF BLEND with partial alpha
        float opacity = 1.0f;      // the base colour factor's alpha (see-through materials)
        bool transmissive = false; // KHR_materials_transmission: drawn over a copy of the scene
        // GpuObject's material extension fields (thickness and attenuation distance in model units).
        glm::vec4 uvTransformU{1.0f, 0.0f, 0.0f, 1.5f};
        glm::vec4 uvTransformV{0.0f, 1.0f, 0.0f, 1.0f};
        glm::vec4 specularClearcoat{1.0f, 1.0f, 1.0f, 0.0f};
        glm::vec4 sheen{0.0f};
        glm::vec4 transmission{0.0f};
        glm::vec4 attenuation{1.0f, 1.0f, 1.0f, 0.0f};
    };
    [[nodiscard]] ModelSurface modelSurface(AssetLibrary& assets, const std::string& meshAsset);
    /// The same for a material asset (.lmat), which replaces everything else.
    [[nodiscard]] static ModelSurface materialSurface(const MaterialAsset& material);
    /// GPU copies of material graph programs (float textures the shaders interpret).
    struct GraphProgramGpu {
        std::weak_ptr<const MaterialProgram> source;
        nvrhi::TextureHandle fragment, vertex;
        std::uint32_t fragmentIndex = UINT32_MAX, vertexIndex = UINT32_MAX;
        std::uint64_t lastUsed = 0;
    };
    [[nodiscard]] const GraphProgramGpu& graphProgram(const std::shared_ptr<const MaterialProgram>& program);
    void releaseGraphPrograms(bool all);
    std::unordered_map<const MaterialProgram*, GraphProgramGpu> graphPrograms_;
    std::vector<GraphProgramGpu> retiredGraphPrograms_;
    std::unordered_map<std::string, ModelSurface> modelSurfaces_;
    const AssetLibrary* surfaceAssets_ = nullptr;
    std::uint64_t surfaceGeneration_ = 0;
    [[nodiscard]] const struct GpuMesh* meshFor(const MeshRendererComponent& mesh, AssetLibrary* assets, bool block);
    /// Pipelines that draw into the output image, one per output format.
    nvrhi::IGraphicsPipeline* outputPipeline(std::unordered_map<int, nvrhi::GraphicsPipelineHandle>& cache,
                                             const PipelineDesc& desc, nvrhi::Format format);

    RenderContext& ctx_;
    nvrhi::GraphicsPipelineHandle sky_;
    nvrhi::GraphicsPipelineHandle opaque_;
    nvrhi::GraphicsPipelineHandle opaqueSkinned_;
    nvrhi::GraphicsPipelineHandle opaqueTwoSided_;
    // See-through surfaces: blended over the shaded scene, sorted back to front, no depth writes.
    nvrhi::GraphicsPipelineHandle transparent_;
    nvrhi::GraphicsPipelineHandle transparentTwoSided_;
    nvrhi::GraphicsPipelineHandle transparentSkinned_;
    nvrhi::GraphicsPipelineHandle grid_;
    nvrhi::GraphicsPipelineHandle particlesAlpha_;
    nvrhi::GraphicsPipelineHandle particlesAdditive_;
    nvrhi::GraphicsPipelineHandle shadow_;
    nvrhi::GraphicsPipelineHandle shadowSkinned_;
    nvrhi::GraphicsPipelineHandle shadowMasked_;  // alpha-tested casters (no culling, like all shadows)
    nvrhi::GraphicsPipelineHandle vsmClear_;
    // Temporal anti-aliasing: the jitter sequence, last frame's matrices (camera and per object,
    // by entity) and two history images that swap every frame.
    nvrhi::ComputePipelineHandle taa_;
    nvrhi::ComputePipelineHandle motionBlur_;
    nvrhi::ComputePipelineHandle contactShadows_;
    nvrhi::ComputePipelineHandle exposureHistogram_;
    nvrhi::ComputePipelineHandle exposureAdapt_;
    nvrhi::TextureHandle exposure_;  // R32F 1x1: the adapted exposure, kept across frames
    std::uint32_t exposureIndex_ = UINT32_MAX;
    double exposureTime_ = -1.0;     // when it was last adapted (< 0: start from the target)
    nvrhi::ComputePipelineHandle fogInject_;
    nvrhi::ComputePipelineHandle fogIntegrate_;
    nvrhi::ComputePipelineHandle ssrTrace_;
    nvrhi::ComputePipelineHandle ssrResolve_;
    // The opaque pipelines writing normal + roughness and the specular weight too (for SSR).
    nvrhi::GraphicsPipelineHandle opaqueReflections_;
    nvrhi::GraphicsPipelineHandle opaqueReflectionsTwoSided_;
    nvrhi::GraphicsPipelineHandle opaqueReflectionsSkinned_;
    std::uint64_t taaFrame_ = 0;
    bool taaHistoryValid_ = false;
    glm::mat4 previousViewProjection_{1.0f};
    glm::vec2 previousJitter_{0.0f};
    glm::vec3 previousCameraPosition_{0.0f};
    glm::vec3 previousCameraForward_{0.0f, 0.0f, -1.0f};
    std::unordered_map<std::uint64_t, glm::mat4> previousWorlds_;
    std::array<nvrhi::TextureHandle, 2> taaHistory_;
    std::array<std::uint32_t, 2> taaHistoryIndex_{UINT32_MAX, UINT32_MAX};
    std::uint32_t taaWidth_ = 0;
    std::uint32_t taaHeight_ = 0;
    VirtualShadowMap vsm_;
    nvrhi::TextureHandle vsmPool_;       // physical pages (depth), kept across frames
    nvrhi::ComputePipelineHandle vsmMark_;
    /// Spot and point light shadows: a depth atlas of kLocalShadowGrid^2 tiles kept across frames.
    /// A light keeps its tiles while it is among the shadowed ones; a tile is drawn again only
    /// when what it shows could have changed (the light, or a caster in it).
    nvrhi::TextureHandle localAtlas_;
    std::uint32_t localAtlasIndex_ = UINT32_MAX;
    struct LocalAllocation {
        std::uint64_t light = 0;  // identity: the light's position, direction, range, kind
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };
    std::vector<LocalAllocation> localAllocations_;
    std::array<std::uint64_t, 64> localTileContents_{};  // per tile, a hash of what it holds (0 = nothing)
    /// GPU page requests in flight: written by frame n, read by frame n + 2 (done by then).
    struct VsmRequests {
        nvrhi::StagingTextureHandle staging;
        bool pending = false;
        int levels = 0;
        std::array<glm::ivec2, 12> windowMin{};
        std::uint64_t generation = 0;
        glm::mat4 viewProjection{1.0f};  // the view they were made for
    };
    std::array<VsmRequests, 2> vsmRequests_;
    std::uint64_t vsmRequestFrame_ = 0;
    nvrhi::TextureHandle vsmPageTable_;  // RGBA32F, see VsmFrame::pageTable
    std::uint32_t vsmPoolIndex_ = UINT32_MAX;
    std::uint32_t vsmPageTableIndex_ = UINT32_MAX;
    nvrhi::GraphicsPipelineHandle depthPrepass_;
    nvrhi::GraphicsPipelineHandle depthPrepassSkinned_;
    nvrhi::GraphicsPipelineHandle depthPrepassTwoSided_;
    nvrhi::GraphicsPipelineHandle depthPrepassMasked_;
    nvrhi::GraphicsPipelineHandle depthPrepassMaskedTwoSided_;
    nvrhi::GraphicsPipelineHandle depthPrepassDithered_;  // LOD blending: gives up the other level's pixels
    nvrhi::GraphicsPipelineHandle depthPrepassDitheredTwoSided_;
    nvrhi::GraphicsPipelineHandle ssao_;
    nvrhi::GraphicsPipelineHandle ssaoBlur_;
    nvrhi::GraphicsPipelineHandle bloomDown_;
    nvrhi::GraphicsPipelineHandle bloomUp_;
    nvrhi::GraphicsPipelineHandle decal_;
    nvrhi::GraphicsPipelineHandle water_;
    nvrhi::GraphicsPipelineHandle depthOfField_;
    nvrhi::GraphicsPipelineHandle shafts_;
    nvrhi::GraphicsPipelineHandle shaftsAdd_;
    std::unordered_map<int, nvrhi::GraphicsPipelineHandle> tonemap_;
    std::unordered_map<int, nvrhi::GraphicsPipelineHandle> overlay_;
    std::unordered_map<int, nvrhi::GraphicsPipelineHandle> fxaa_;
    std::unordered_map<int, nvrhi::GraphicsPipelineHandle> hud_;
    std::unique_ptr<HudFont> hudFont_;  // built on first use
    struct RetiredAtlas {
        nvrhi::TextureHandle texture;
        std::uint32_t index = UINT32_MAX;
        std::uint64_t frame = 0;
    };
    std::vector<RetiredAtlas> retiredHudAtlases_;  // replaced glyph atlases, released once surely unused
    nvrhi::TextureHandle hudAtlas_;
    std::uint32_t hudAtlasIndex_ = UINT32_MAX;
    nvrhi::TextureHandle spriteAtlas_;
    std::uint32_t spriteAtlasIndex_ = UINT32_MAX;
    const SpriteAtlas* spriteAtlasSource_ = nullptr;
    std::uint64_t spriteAtlasVersion_ = 0;
    /// Terrains and foliage: how they look (as a Mesh Renderer would) and their copies' world
    /// matrices, rebuilt when their key changes; dropped when unused for a while.
    struct LandscapeEntry {
        std::uint64_t key = 0;
        MeshRendererComponent look;
        std::vector<glm::mat4> worlds;
        std::uint64_t lastFrame = 0;
    };
    std::unordered_map<std::uint64_t, LandscapeEntry> landscape_;
    std::uint64_t frameCounter_ = 0;
    /// Baked light probe volumes on the GPU, by absolute file path.
    struct ProbeVolumeGpu {
        nvrhi::TextureHandle texture;
        std::uint32_t index = UINT32_MAX;
        glm::uvec3 counts{0u};
        std::uint64_t lastFrame = 0;
        bool failed = false;
    };
    std::unordered_map<std::string, ProbeVolumeGpu> probeVolumes_;
    [[nodiscard]] const ProbeVolumeGpu* probeVolume(AssetLibrary& assets, const std::string& key);

    [[nodiscard]] GpuOcclusionCuller* ensureGpuCulling();
    [[nodiscard]] SdfGiPass* ensureSdfGi();
    [[nodiscard]] RestirPass* ensureRestir();
    void ensureSdfGiVolume(const Scene& scene);
    void ensureSdfGiOutput(std::uint32_t width, std::uint32_t height);

    std::unique_ptr<GpuOcclusionCuller> gpuCulling_;
    std::unique_ptr<SdfGiPass> sdfGi_;
    std::unique_ptr<RestirPass> restir_;
    bool gpuCullingFailed_ = false;
    bool sdfGiFailed_ = false;
    bool restirFailed_ = false;
    bool gpuCullingActiveLastFrame_ = false;
    const Scene* gpuCullingLastScene_ = nullptr;
    glm::vec3 gpuCullingPreviousCameraPosition_{0.0f};
    glm::vec3 gpuCullingPreviousCameraForward_{0.0f, 0.0f, -1.0f};
    bool sdfGiAutoBuilt_ = false;
    std::uint64_t sdfGiSceneHash_ = 0;
    nvrhi::TextureHandle sdfGiOutput_;
    std::uint32_t sdfGiOutputIndex_ = UINT32_MAX;
    std::uint32_t sdfGiWidth_ = 0;
    std::uint32_t sdfGiHeight_ = 0;
    bool restirActiveLastFrame_ = false;
    const Scene* restirLastScene_ = nullptr;
    glm::vec3 restirPreviousCameraPosition_{0.0f};
    glm::vec3 restirPreviousCameraForward_{0.0f, 0.0f, -1.0f};
};

} // namespace lumora::render
