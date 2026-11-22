#include "runtime/AudioWorld.hpp"

#include "core/LockFreeQueue.hpp"
#include "core/Log.hpp"
#include "runtime/PhysicsWorld.hpp"
#include "scene/Scene.hpp"

#include <glm/geometric.hpp>
#include <glm/matrix.hpp>

#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace lumora {

namespace {
/// Clips larger than this stream from disk instead of being decoded up front.
constexpr std::uintmax_t kStreamAboveBytes = 2u * 1024u * 1024u;
/// Nominal fixed simulation step for velocity estimation when no rigid-body velocity is available.
constexpr float kNominalAudioStepSec = 1.0f / 60.0f;

struct VoiceDsp;

/// Standard-layout wrapper so `reinterpret_cast<VoiceNodeWrapper*>(pNode)` is well-defined.
struct VoiceNodeWrapper {
    ma_node_base base{};
    VoiceDsp* owner = nullptr;
};

struct VoiceDspCommand {
    std::uint64_t seq = 0;
    float lpfCutoffHz = 20000.0f;
    bool active = false;
};

struct VoiceDsp {
    VoiceNodeWrapper node{};
    BiquadLowPassFilter lpf{};
    SpscRingBuffer<VoiceDspCommand, 128> cmdQueue{};
    std::atomic<std::uint64_t> fallbackSeq{0};
    std::atomic<float> fallbackCutoffHz{20000.0f};
    std::atomic<bool> fallbackActive{false};
    std::uint64_t nextSeq = 0;
    std::uint64_t lastConsumedSeq = 0;
    bool audioThreadActive = false;
    bool nodeInitialized = false;
};

void onVoiceDspProcess(ma_node* pNode, const float** /*ppFramesIn*/, ma_uint32* /*pFrameCountIn*/,
                       float** ppFramesOut, ma_uint32* pFrameCountOut) {
    auto* wrapper = reinterpret_cast<VoiceNodeWrapper*>(pNode);
    if (wrapper == nullptr || wrapper->owner == nullptr || ppFramesOut == nullptr || ppFramesOut[0] == nullptr ||
        pFrameCountOut == nullptr || *pFrameCountOut == 0u) {
        return;
    }
    VoiceDsp& dsp = *wrapper->owner;
    VoiceDspCommand cmd;
    while (dsp.cmdQueue.tryPop(cmd)) {
        if (cmd.seq > dsp.lastConsumedSeq) {
            dsp.lastConsumedSeq = cmd.seq;
            dsp.audioThreadActive = cmd.active;
            dsp.lpf.setCutoff(cmd.lpfCutoffHz);
        }
    }
    const std::uint64_t fbSeq = dsp.fallbackSeq.load(std::memory_order_acquire);
    if (fbSeq > dsp.lastConsumedSeq) {
        dsp.lastConsumedSeq = fbSeq;
        dsp.audioThreadActive = dsp.fallbackActive.load(std::memory_order_relaxed);
        dsp.lpf.setCutoff(dsp.fallbackCutoffHz.load(std::memory_order_relaxed));
    }
    if (dsp.audioThreadActive) {
        dsp.lpf.processInterleavedStereo(ppFramesOut[0], *pFrameCountOut);
    }
}

constexpr ma_node_vtable kVoiceDspVtable = {
    onVoiceDspProcess,
    nullptr,
    1,
    1,
    MA_NODE_FLAG_PASSTHROUGH,
};

void pushVoiceDspCommand(VoiceDsp& dsp, float cutoffHz, bool active) {
    const std::uint64_t seq = ++dsp.nextSeq;
    const VoiceDspCommand cmd{seq, cutoffHz, active};
    if (!dsp.cmdQueue.tryPush(cmd)) {
        dsp.fallbackCutoffHz.store(cutoffHz, std::memory_order_relaxed);
        dsp.fallbackActive.store(active, std::memory_order_relaxed);
        dsp.fallbackSeq.store(seq, std::memory_order_release);
    }
}

std::optional<AcousticHit> raycastAcoustic(const PhysicsWorld& physics, const Scene& scene,
                                           entt::entity ignoreEntity, const glm::vec3& origin,
                                           const glm::vec3& dir, float maxDistance) {
    const float len = glm::length(dir);
    if (len <= 1e-6f || maxDistance <= 0.0f) return std::nullopt;
    const glm::vec3 unitDir = dir / len;
    float traveled = 0.0f;
    const auto& reg = scene.registry();
    for (int hop = 0; hop < 4 && traveled < maxDistance; ++hop) {
        const float remaining = maxDistance - traveled;
        const glm::vec3 rayOrigin = origin + unitDir * traveled;
        const auto hit = physics.raycast(rayOrigin, unitDir, remaining);
        if (!hit) return std::nullopt;
        const bool isTrigger =
            hit->entity != entt::null && reg.valid(hit->entity) &&
            reg.all_of<ColliderComponent>(hit->entity) &&
            reg.get<ColliderComponent>(hit->entity).isTrigger;
        if (hit->entity == ignoreEntity || isTrigger) {
            float advance = std::max(hit->distance, 0.0f) + 0.02f;
            if (hit->distance <= 1e-3f && hit->entity != entt::null && reg.valid(hit->entity)) {
                if (const auto* col = reg.try_get<ColliderComponent>(hit->entity)) {
                    const glm::vec3 ext = col->shape == ColliderShape::Box
                                              ? col->size * 0.5f
                                              : glm::vec3(std::max(col->radius, col->height * 0.5f));
                    advance = std::max(advance, glm::length(ext) + 0.05f);
                } else {
                    advance = 0.25f;
                }
            }
            traveled += advance;
            continue;
        }
        AcousticHit out;
        out.distance = traveled + hit->distance;
        out.point = hit->point;
        out.normal = hit->normal;
        out.material = defaultAcousticMaterial(AcousticMaterialKind::Concrete);
        return out;
    }
    return std::nullopt;
}
} // namespace

struct AudioWorld::Impl {
    Scene& scene;
    std::filesystem::path root;
    bool device = true;
    ma_engine engine{};
    bool engineReady = false;
    ma_uint32 channels = 2;
    float sampleRate = 48000.0f;
    struct Voice {
        std::unique_ptr<ma_sound> sound;
        std::unique_ptr<VoiceDsp> dsp;
        std::string clip;
        bool started = false;
        AudioBus bus = AudioBus::Sfx;
    };
    std::unordered_map<entt::entity, Voice> voices;
    std::unordered_map<std::string, bool> reported;  // clips that failed to load (logged once)
    // Mixer: Master, then one group per AudioBus below it.
    static constexpr int kBuses = 5;
    ma_sound_group master{};
    ma_sound_group buses[kBuses]{};
    bool groupsReady = false;
    float busVolumes[kBuses + 1] = {1, 1, 1, 1, 1, 1};  // index 0 Master, then AudioBus + 1
    bool busMutedFlags[kBuses + 1] = {};

    // Optional spatial acoustics pipeline (disabled by default).
    const PhysicsWorld* physics = nullptr;
    bool acousticsEnabled = false;
    SpatialAcousticsSolver acoustics{};
    glm::vec3 listenerPos{0.0f};
    glm::vec3 explicitListenerVel{0.0f};
    bool hasExplicitListenerVel = false;
    glm::vec3 prevUpdateListenerPos{0.0f};
    bool hasPrevUpdateListenerPos = false;
    std::unordered_map<entt::entity, glm::vec3> prevEntityPositions;
    std::unordered_map<entt::entity, AcousticSourceResult> lastAcoustics;

    struct ReverbCommand {
        std::uint64_t seq = 0;
        float rt60Sec = 0.4f;
        float wetMix = 0.0f;
        bool active = false;
    };
    SchroederReverbDsp reverb{};
    SpscRingBuffer<ReverbCommand, 256> reverbQueue{};
    std::atomic<std::uint64_t> reverbFallbackSeq{0};
    std::atomic<float> reverbFallbackRt60{0.4f};
    std::atomic<float> reverbFallbackWet{0.0f};
    std::atomic<bool> reverbFallbackActive{false};
    std::uint64_t nextReverbSeq = 0;
    std::uint64_t lastConsumedReverbSeq = 0;
    bool audioThreadReverbActive = false;

    static void onEngineProcess(void* pUserData, float* pFramesOut, ma_uint64 frameCount) {
        auto* self = static_cast<Impl*>(pUserData);
        if (self == nullptr || pFramesOut == nullptr || frameCount == 0u || self->channels != 2u) {
            return;
        }
        ReverbCommand cmd;
        while (self->reverbQueue.tryPop(cmd)) {
            if (cmd.seq > self->lastConsumedReverbSeq) {
                self->lastConsumedReverbSeq = cmd.seq;
                self->audioThreadReverbActive = cmd.active;
                self->reverb.setParameters(cmd.rt60Sec, cmd.wetMix);
            }
        }
        const std::uint64_t fbSeq = self->reverbFallbackSeq.load(std::memory_order_acquire);
        if (fbSeq > self->lastConsumedReverbSeq) {
            self->lastConsumedReverbSeq = fbSeq;
            self->audioThreadReverbActive = self->reverbFallbackActive.load(std::memory_order_relaxed);
            self->reverb.setParameters(self->reverbFallbackRt60.load(std::memory_order_relaxed),
                                       self->reverbFallbackWet.load(std::memory_order_relaxed));
        }
        if (!self->audioThreadReverbActive) {
            return;
        }
        ma_uint64 remaining = frameCount;
        float* cursor = pFramesOut;
        while (remaining > 0u) {
            const auto chunk = static_cast<std::uint32_t>(
                std::min<ma_uint64>(remaining, static_cast<ma_uint64>(std::numeric_limits<std::uint32_t>::max())));
            self->reverb.processInterleavedStereo(cursor, chunk);
            cursor += static_cast<std::size_t>(chunk) * 2u;
            remaining -= chunk;
        }
    }

    void pushReverb(float rt60Sec, float wetMix, bool active) {
        const std::uint64_t seq = ++nextReverbSeq;
        const ReverbCommand cmd{seq, rt60Sec, wetMix, active};
        if (!reverbQueue.tryPush(cmd)) {
            reverbFallbackRt60.store(rt60Sec, std::memory_order_relaxed);
            reverbFallbackWet.store(wetMix, std::memory_order_relaxed);
            reverbFallbackActive.store(active, std::memory_order_relaxed);
            reverbFallbackSeq.store(seq, std::memory_order_release);
        }
    }

    bool initGroups() {
        if (ma_sound_group_init(&engine, 0, nullptr, &master) != MA_SUCCESS) return false;
        for (int i = 0; i < kBuses; ++i) {
            if (ma_sound_group_init(&engine, 0, &master, &buses[i]) != MA_SUCCESS) {
                for (int j = 0; j < i; ++j) ma_sound_group_uninit(&buses[j]);
                ma_sound_group_uninit(&master);
                return false;
            }
        }
        groupsReady = true;
        return true;
    }
    ma_sound_group* group(AudioBus bus) {
        const int i = static_cast<int>(bus);
        return groupsReady && i >= 0 && i < kBuses ? &buses[i] : nullptr;
    }
    void applyBus(int index) {
        if (!groupsReady) return;
        ma_sound_group* g = index == 0 ? &master : &buses[index - 1];
        ma_sound_group_set_volume(g, busMutedFlags[index] ? 0.0f : busVolumes[index]);
    }

    explicit Impl(Scene& s) : scene(s) {}
    void uninitVoice(Voice& voice) {
        if (voice.sound) ma_sound_uninit(voice.sound.get());
        if (voice.dsp && voice.dsp->nodeInitialized) {
            ma_node_uninit(&voice.dsp->node.base, nullptr);
            voice.dsp->nodeInitialized = false;
        }
    }
    ~Impl() {
        for (auto& [entity, voice] : voices) uninitVoice(voice);
        voices.clear();
        if (groupsReady) {
            for (auto& g : buses) ma_sound_group_uninit(&g);
            ma_sound_group_uninit(&master);
        }
        if (engineReady) ma_engine_uninit(&engine);
    }

    void apply(Voice& voice, const AudioSourceComponent& source) {
        ma_sound* s = voice.sound.get();
        ma_sound_set_volume(s, std::max(source.volume, 0.0f));
        ma_sound_set_pitch(s, std::max(source.pitch, 0.01f));
        ma_sound_set_looping(s, source.loop ? MA_TRUE : MA_FALSE);
        ma_sound_set_spatialization_enabled(s, source.spatial ? MA_TRUE : MA_FALSE);
        ma_sound_set_attenuation_model(s, ma_attenuation_model_inverse);
        ma_sound_set_min_distance(s, std::max(source.minDistance, 0.01f));
        ma_sound_set_max_distance(s, std::max(source.maxDistance, source.minDistance + 0.01f));
    }

    bool createVoice(entt::entity e, const AudioSourceComponent& source) {
        if (source.clip.empty()) return false;
        const std::filesystem::path relative = std::filesystem::path(source.clip).lexically_normal();
        if (relative.is_absolute() || relative.empty() || *relative.begin() == "..") {
            if (!reported[source.clip]) log::warn("Audio", "clip '{}' must be inside the project", source.clip);
            reported[source.clip] = true;
            return false;
        }
        const std::filesystem::path file = root / relative;
        std::error_code ec;
        const auto size = std::filesystem::file_size(file, ec);
        ma_uint32 flags = ec || size > kStreamAboveBytes ? MA_SOUND_FLAG_STREAM : MA_SOUND_FLAG_DECODE;
        std::unique_ptr<VoiceDsp> dsp;
        if (channels == 2u) {
            dsp = std::make_unique<VoiceDsp>();
            dsp->node.owner = dsp.get();
            dsp->lpf.prepare(sampleRate);
            ma_node_config nodeCfg = ma_node_config_init();
            nodeCfg.vtable = &kVoiceDspVtable;
            nodeCfg.pInputChannels = &channels;
            nodeCfg.pOutputChannels = &channels;
            if (ma_node_init(ma_engine_get_node_graph(&engine), &nodeCfg, nullptr, &dsp->node.base) == MA_SUCCESS) {
                dsp->nodeInitialized = true;
                ma_node* target = group(source.bus) != nullptr ? static_cast<ma_node*>(group(source.bus))
                                                                : ma_engine_get_endpoint(&engine);
                if (ma_node_attach_output_bus(&dsp->node.base, 0, target, 0) != MA_SUCCESS) {
                    ma_node_uninit(&dsp->node.base, nullptr);
                    dsp->nodeInitialized = false;
                    dsp.reset();
                }
            } else {
                dsp.reset();
            }
        }
        auto sound = std::make_unique<ma_sound>();
        const std::string fileStr = file.string();
        ma_result result = MA_ERROR;
        if (dsp && dsp->nodeInitialized) {
            ma_sound_config cfg = ma_sound_config_init_2(&engine);
            cfg.pFilePath = fileStr.c_str();
            cfg.flags = flags;
            cfg.pInitialAttachment = &dsp->node.base;
            cfg.initialAttachmentInputBusIndex = 0;
            result = ma_sound_init_ex(&engine, &cfg, sound.get());
        } else {
            result = ma_sound_init_from_file(&engine, fileStr.c_str(), flags, group(source.bus), nullptr, sound.get());
        }
        if (result != MA_SUCCESS) {
            if (dsp && dsp->nodeInitialized) {
                ma_node_uninit(&dsp->node.base, nullptr);
                dsp->nodeInitialized = false;
            }
            if (!reported[source.clip]) {
                log::warn("Audio", "cannot play '{}': {}", source.clip, ma_result_description(result));
            }
            reported[source.clip] = true;
            return false;
        }
        Voice voice{std::move(sound), std::move(dsp), source.clip, false, source.bus};
        apply(voice, source);
        voices.emplace(e, std::move(voice));
        return true;
    }

    void updateVoices(bool runAcoustics) {
        auto& reg = scene.registry();
        // Voices of removed entities (or removed sources) stop.
        for (auto it = voices.begin(); it != voices.end();) {
            if (!reg.valid(it->first) || !reg.all_of<AudioSourceComponent>(it->first)) {
                uninitVoice(it->second);
                it = voices.erase(it);
            } else {
                ++it;
            }
        }
        std::erase_if(prevEntityPositions, [&](const auto& entry) {
            return !reg.valid(entry.first) || !reg.all_of<AudioSourceComponent>(entry.first);
        });
        std::erase_if(lastAcoustics, [&](const auto& entry) {
            return !reg.valid(entry.first) || !reg.all_of<AudioSourceComponent>(entry.first);
        });

        const bool evalAcoustics = runAcoustics && acousticsEnabled && physics != nullptr;
        glm::vec3 listenerVel{0.0f};
        EnvironmentReverbResult envReverb{};
        bool envEvaluated = false;

        if (evalAcoustics) {
            if (hasExplicitListenerVel) {
                listenerVel = explicitListenerVel;
            } else if (hasPrevUpdateListenerPos) {
                listenerVel = (listenerPos - prevUpdateListenerPos) / kNominalAudioStepSec;
            }
            prevUpdateListenerPos = listenerPos;
            hasPrevUpdateListenerPos = true;

            for (auto [e, source] : reg.view<AudioSourceComponent>().each()) {
                if (!source.spatial) continue;
                const AcousticRaycastFn envRay = [this](const glm::vec3& o, const glm::vec3& d, float maxD) {
                    return raycastAcoustic(*physics, scene, entt::null, o, d, maxD);
                };
                envReverb = acoustics.estimateEnvironmentReverb(listenerPos, envRay);
                envEvaluated = true;
                pushReverb(envReverb.rt60Sec, envReverb.wetMix, envReverb.wetMix > 1e-4f);
                break;
            }
            if (!envEvaluated) {
                pushReverb(0.4f, 0.0f, false);
            }
        } else if (runAcoustics) {
            lastAcoustics.clear();
            pushReverb(0.4f, 0.0f, false);
        }

        for (auto [e, source] : reg.view<AudioSourceComponent>().each()) {
            auto it = voices.find(e);
            if (it == voices.end()) {
                if (wCreateVoice(e, source)) it = voices.find(e);
            } else if (it->second.clip != source.clip || it->second.bus != source.bus) {
                uninitVoice(it->second);
                voices.erase(it);
                if (wCreateVoice(e, source)) {
                    it = voices.find(e);
                } else {
                    it = voices.end();
                }
            }

            const glm::vec3 p(scene.worldMatrix(e)[3]);
            if (it != voices.end()) {
                Voice& voice = it->second;
                apply(voice, source);
                if (source.spatial) {
                    ma_sound_set_position(voice.sound.get(), p.x, p.y, p.z);
                }
                if (!voice.started) {
                    voice.started = true;
                    if (source.playOnStart) ma_sound_start(voice.sound.get());
                }
            }

            if (evalAcoustics && source.spatial) {
                glm::vec3 sourceVel = physics->linearVelocity(e);
                const auto prevIt = prevEntityPositions.find(e);
                if (glm::dot(sourceVel, sourceVel) <= 1e-10f && prevIt != prevEntityPositions.end()) {
                    sourceVel = (p - prevIt->second) / kNominalAudioStepSec;
                }
                prevEntityPositions[e] = p;

                const AcousticRaycastFn srcRay = [this, e](const glm::vec3& o, const glm::vec3& d, float maxD) {
                    return raycastAcoustic(*physics, scene, e, o, d, maxD);
                };
                const AcousticSourceResult res = acoustics.evaluateSource(
                    p, sourceVel, listenerPos, listenerVel, srcRay, envReverb, source.minDistance, source.maxDistance);
                lastAcoustics[e] = res;

                if (it != voices.end()) {
                    Voice& voice = it->second;
                    const float occGain = std::clamp(1.0f - res.occlusion, 0.0f, 1.0f);
                    ma_sound_set_volume(voice.sound.get(), std::max(source.volume, 0.0f) * occGain);
                    ma_sound_set_pitch(voice.sound.get(),
                                       std::max(source.pitch, 0.01f) * std::clamp(res.dopplerPitch, 0.5f, 2.0f));
                    if (voice.dsp) {
                        pushVoiceDspCommand(*voice.dsp, res.lpfCutoffHz, true);
                    }
                }
            } else {
                prevEntityPositions[e] = p;
                lastAcoustics.erase(e);
                if (it != voices.end() && it->second.dsp) {
                    pushVoiceDspCommand(*it->second.dsp, 20000.0f, false);
                }
            }
        }
    }

    bool wCreateVoice(entt::entity e, const AudioSourceComponent& source) { return createVoice(e, source); }
};

AudioWorld::AudioWorld(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AudioWorld::~AudioWorld() = default;

bool AudioWorld::usesDevice() const noexcept { return impl_->device; }

Result<std::unique_ptr<AudioWorld>> AudioWorld::create(Scene& scene, const Options& options) {
    auto impl = std::make_unique<Impl>(scene);
    impl->root = options.projectRoot;
    impl->device = options.device;
    ma_engine_config config = ma_engine_config_init();
    config.onProcess = &Impl::onEngineProcess;
    config.pProcessUserData = impl.get();
    if (!options.device) {
        config.noDevice = MA_TRUE;
        config.channels = 2;
        config.sampleRate = 48000;
    }
    const ma_result result = ma_engine_init(&config, &impl->engine);
    if (result != MA_SUCCESS) return makeError("audio unavailable: {}", ma_result_description(result));
    impl->engineReady = true;
    impl->channels = ma_engine_get_channels(&impl->engine);
    impl->sampleRate = static_cast<float>(ma_engine_get_sample_rate(&impl->engine));
    impl->reverb.prepare(impl->sampleRate);
    if (!impl->initGroups()) log::warn("Audio", "mixer buses unavailable; sounds play straight to the output");
    log::info("Audio", "audio engine started ({} Hz{})", ma_engine_get_sample_rate(&impl->engine),
              options.device ? "" : ", no device");
    return std::unique_ptr<AudioWorld>(new AudioWorld(std::move(impl)));
}

namespace {
int busIndex(std::string_view name) {
    std::string lower(name);
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "master") return 0;
    if (lower == "music") return 1;
    if (lower == "sfx") return 2;
    if (lower == "ui") return 3;
    if (lower == "voice") return 4;
    if (lower == "ambience") return 5;
    return -1;
}
} // namespace

bool AudioWorld::setBusVolume(std::string_view bus, float volume) {
    const int i = busIndex(bus);
    if (i < 0) return false;
    impl_->busVolumes[i] = std::clamp(volume, 0.0f, 4.0f);
    impl_->applyBus(i);
    return true;
}

float AudioWorld::busVolume(std::string_view bus) const {
    const int i = busIndex(bus);
    return i < 0 ? -1.0f : impl_->busVolumes[i];
}

bool AudioWorld::setBusMuted(std::string_view bus, bool muted) {
    const int i = busIndex(bus);
    if (i < 0) return false;
    impl_->busMutedFlags[i] = muted;
    impl_->applyBus(i);
    return true;
}

bool AudioWorld::busMuted(std::string_view bus) const {
    const int i = busIndex(bus);
    return i >= 0 && impl_->busMutedFlags[i];
}

void AudioWorld::setPhysics(const PhysicsWorld* physics) { impl_->physics = physics; }

void AudioWorld::setAcousticsEnabled(bool enabled) {
    impl_->acousticsEnabled = enabled;
    if (!enabled) {
        impl_->lastAcoustics.clear();
        impl_->pushReverb(0.4f, 0.0f, false);
        for (auto& [entity, voice] : impl_->voices) {
            (void)entity;
            if (voice.dsp) pushVoiceDspCommand(*voice.dsp, 20000.0f, false);
        }
    }
}

bool AudioWorld::acousticsEnabled() const noexcept { return impl_->acousticsEnabled; }

std::optional<AcousticSourceResult> AudioWorld::lastAcousticsFor(entt::entity entity) const {
    const auto it = impl_->lastAcoustics.find(entity);
    if (it == impl_->lastAcoustics.end()) return std::nullopt;
    return it->second;
}

bool AudioWorld::sceneHasAudio(const Scene& scene) {
    return !scene.registry().view<AudioSourceComponent>().empty();
}

void AudioWorld::update() { impl_->updateVoices(true); }

void AudioWorld::setListener(const glm::vec3& position, const glm::vec3& forward, const glm::vec3& up) {
    impl_->listenerPos = position;
    ma_engine_listener_set_position(&impl_->engine, 0, position.x, position.y, position.z);
    ma_engine_listener_set_direction(&impl_->engine, 0, forward.x, forward.y, forward.z);
    ma_engine_listener_set_world_up(&impl_->engine, 0, up.x, up.y, up.z);
}

void AudioWorld::setListenerFromView(const glm::mat4& view) {
    const glm::mat4 camera = glm::inverse(view);
    setListener(glm::vec3(camera[3]), -glm::vec3(camera[2]), glm::vec3(camera[1]));
}

void AudioWorld::setListenerVelocity(const glm::vec3& velocity) {
    impl_->explicitListenerVel = velocity;
    impl_->hasExplicitListenerVel = true;
    ma_engine_listener_set_velocity(&impl_->engine, 0, velocity.x, velocity.y, velocity.z);
}

void AudioWorld::setMasterVolume(float volume) {
    ma_engine_set_volume(&impl_->engine, std::clamp(volume, 0.0f, 1.0f));
}

void AudioWorld::setPaused(bool paused) {
    if (!impl_->device) return;
    if (paused) {
        ma_engine_stop(&impl_->engine);
    } else {
        ma_engine_start(&impl_->engine);
    }
}

bool AudioWorld::play(entt::entity entity) {
    impl_->updateVoices(false);  // a source added this step gets its voice now
    const auto it = impl_->voices.find(entity);
    if (it == impl_->voices.end()) return false;
    it->second.started = true;
    ma_sound_seek_to_pcm_frame(it->second.sound.get(), 0);
    return ma_sound_start(it->second.sound.get()) == MA_SUCCESS;
}

bool AudioWorld::stop(entt::entity entity) {
    const auto it = impl_->voices.find(entity);
    if (it == impl_->voices.end()) return false;
    return ma_sound_stop(it->second.sound.get()) == MA_SUCCESS;
}

bool AudioWorld::isPlaying(entt::entity entity) const {
    const auto it = impl_->voices.find(entity);
    if (it == impl_->voices.end()) return false;
    ma_sound* sound = it->second.sound.get();
    // A finished non-looping sound reports "at end" before its playing flag settles.
    return ma_sound_is_playing(sound) == MA_TRUE && ma_sound_at_end(sound) == MA_FALSE;
}

float AudioWorld::cursorSeconds(entt::entity entity) const {
    const auto it = impl_->voices.find(entity);
    if (it == impl_->voices.end()) return 0.0f;
    float seconds = 0.0f;
    ma_sound_get_cursor_in_seconds(it->second.sound.get(), &seconds);
    return seconds;
}

std::size_t AudioWorld::voiceCount() const { return impl_->voices.size(); }

float AudioWorld::mix(std::uint32_t frames) {
    std::vector<float> buffer(static_cast<std::size_t>(frames) * 2);
    ma_uint64 read = 0;
    ma_engine_read_pcm_frames(&impl_->engine, buffer.data(), frames, &read);
    float peak = 0.0f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(read) * 2; ++i) peak = std::max(peak, std::abs(buffer[i]));
    return peak;
}

} // namespace lumora
