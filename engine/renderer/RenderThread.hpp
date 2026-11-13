#pragma once

#include "renderer/Overlay.hpp"
#include "renderer/SceneRenderer.hpp"
#include "renderer/Sprites.hpp"
#include "renderer/vulkan/Swapchain.hpp"
#include "scene/DebugDraw.hpp"
#include "scene/Hud.hpp"
#include "scene/Scene.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lumora::render {

/// What one frame of a viewport needs, copied on the UI thread so another thread can draw it
/// while the scene goes on changing: the components the renderer reads (same entities, same
/// order, so the frame is the one the live scene would give), the camera, the settings and the
/// overlays they point at.
///
/// A component the renderer starts to read must be added to RenderSnapshot.cpp's list (the
/// RenderThread tests compare a snapshot's frame with the live scene's).
struct RenderSnapshot {
    Scene scene{"Snapshot"};
    CameraView camera;
    RenderSettings settings;  // its pointers lead to the copies below (or are null)
    OverlayGeometry overlay;
    SpriteBatch sprites;      // the atlas itself is shared (it does not change while in use)
    HudCanvas hud;
    DebugDraw debug;
    VkExtent2D extent{};      // of the target
    rhi::PresentPolicy policy = rhi::PresentPolicy::Vsync;

    /// Fills this from the live state, reusing its storage.
    void capture(const Scene& live, const CameraView& view, const RenderSettings& liveSettings);
};

/// Snapshots to fill, recycled between frames (thread-safe).
class SnapshotPool {
public:
    [[nodiscard]] std::unique_ptr<RenderSnapshot> take();
    void give(std::unique_ptr<RenderSnapshot> snapshot);

private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<RenderSnapshot>> free_;
};

/// Renders viewports' frames on a thread of its own, so the UI thread never waits for the GPU
/// (fences of earlier frames, swapchain images, presentation): it hands each frame over and goes
/// on with events and the simulation. Work is keyed by its viewport; work that has not started is
/// replaced by newer work of the same key (a slow GPU shows the latest state, the UI does not
/// queue up frames). Rendering takes the Renderer's lock, so captures and other renderer calls
/// from the UI thread wait for the frame in progress, never the other way round.
class RenderThread {
public:
    using Work = std::move_only_function<void()>;

    RenderThread();
    ~RenderThread();  // finishes the work in progress; drops the rest
    RenderThread(const RenderThread&) = delete;
    RenderThread& operator=(const RenderThread&) = delete;

    /// Queues `work` for `key`; replaces queued work of that key.
    void post(const void* key, Work work);
    /// Returns once nothing of `key` is queued or running (queued work is dropped). Call it
    /// before the key's target is resized away or destroyed.
    void forget(const void* key);
    /// Nothing of `key` queued or running.
    [[nodiscard]] bool idle(const void* key) const;
    /// False when LUMORA_RENDER_THREAD=0: viewports then render on the UI thread as before.
    [[nodiscard]] static bool enabledByEnvironment();

private:
    void loop(std::stop_token stop);

    struct Job {
        const void* key;
        Work work;
    };
    mutable std::mutex mutex_;
    std::condition_variable_any changed_;  // work arrived, or work finished
    std::deque<Job> queue_;
    const void* running_ = nullptr;
    std::jthread thread_;
};

} // namespace lumora::render
