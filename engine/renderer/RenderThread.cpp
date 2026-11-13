#include "renderer/RenderThread.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace lumora::render {

void RenderSnapshot::capture(const Scene& live, const CameraView& view, const RenderSettings& liveSettings) {
    // Everything SceneRenderer and its helpers (landscape, skinning) read from a scene.
    live.copyInto<IdComponent, NameComponent, TransformComponent, WorldTransformComponent, HierarchyComponent,
                  MeshRendererComponent, LightComponent, CameraComponent, DecalComponent, TerrainComponent, WaterComponent,
                  ReflectionProbeComponent, LightProbeVolumeComponent, FoliageComponent, ParticleEmitterComponent,
                  ParticleStateComponent, SkinComponent, MorphWeightsComponent, ModelNodeComponent>(scene);
    camera = view;
    settings = liveSettings;
    auto copy = [](const auto* from, auto& to) -> decltype(from) {
        if (from == nullptr) return nullptr;
        to = *from;
        return &to;
    };
    settings.overlay = copy(liveSettings.overlay, overlay);
    settings.sprites = copy(liveSettings.sprites, sprites);
    settings.hud = copy(liveSettings.hud, hud);
    settings.debug = copy(liveSettings.debug, debug);
}

std::unique_ptr<RenderSnapshot> SnapshotPool::take() {
    std::scoped_lock lock(mutex_);
    if (free_.empty()) return std::make_unique<RenderSnapshot>();
    std::unique_ptr<RenderSnapshot> snapshot = std::move(free_.back());
    free_.pop_back();
    return snapshot;
}

void SnapshotPool::give(std::unique_ptr<RenderSnapshot> snapshot) {
    if (!snapshot) return;
    std::scoped_lock lock(mutex_);
    if (free_.size() < 4) free_.push_back(std::move(snapshot));
}

RenderThread::RenderThread() : thread_([this](std::stop_token stop) { loop(stop); }) {}

RenderThread::~RenderThread() {
    std::deque<Job> dropped;  // destroyed outside the lock (what they hold may take time to free)
    {
        std::scoped_lock lock(mutex_);
        dropped.swap(queue_);
    }
    thread_.request_stop();
    changed_.notify_all();
    thread_.join();
}

bool RenderThread::enabledByEnvironment() {
    const char* value = std::getenv("LUMORA_RENDER_THREAD");
    return value == nullptr || std::string_view(value) != "0";
}

void RenderThread::post(const void* key, Work work) {
    Work replaced;  // destroyed outside the lock
    {
        std::scoped_lock lock(mutex_);
        const auto queued = std::ranges::find(queue_, key, &Job::key);
        if (queued != queue_.end()) {
            replaced = std::exchange(queued->work, std::move(work));  // the newer frame replaces the one not started
        } else {
            queue_.push_back({key, std::move(work)});
        }
    }
    changed_.notify_all();
}

void RenderThread::forget(const void* key) {
    std::vector<Job> dropped;  // destroyed outside the lock
    std::unique_lock lock(mutex_);
    for (auto it = queue_.begin(); it != queue_.end();) {
        if (it->key == key) {
            dropped.push_back(std::move(*it));
            it = queue_.erase(it);
        } else {
            ++it;
        }
    }
    changed_.wait(lock, [&] { return running_ != key; });
    lock.unlock();
}

bool RenderThread::idle(const void* key) const {
    std::scoped_lock lock(mutex_);
    return running_ != key && std::ranges::find(queue_, key, &Job::key) == queue_.end();
}

void RenderThread::loop(std::stop_token stop) {
    log::debug("Renderer", "render thread started");
    std::unique_lock lock(mutex_);
    while (true) {
        changed_.wait(lock, stop, [&] { return !queue_.empty(); });
        if (stop.stop_requested()) break;
        Job job = std::move(queue_.front());
        queue_.pop_front();
        running_ = job.key;
        lock.unlock();
        try {
            job.work();
        } catch (const std::exception& e) {
            log::error("Renderer", "render thread: {}", e.what());  // work reports its own failures
        }
        job.work = nullptr;  // what it captured goes before the key is free again
        lock.lock();
        running_ = nullptr;
        changed_.notify_all();
    }
    log::debug("Renderer", "render thread stopped");
}

} // namespace lumora::render
