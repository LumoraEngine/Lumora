#include "renderer/TextureCache.hpp"

#include "assets/AssetLibrary.hpp"
#include "core/JobSystem.hpp"
#include "core/Log.hpp"
#include "core/Timer.hpp"
#include "renderer/RenderContext.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <span>

namespace lumora::render {

namespace {

constexpr std::uint64_t kRetireFrames = 4;       // frames in flight, and then some
constexpr std::uint32_t kFirstAppearance = 256;  // texels across a texture shows at first
constexpr std::uint64_t kUnusedFrames = 120;     // then it may be dropped for the budget

/// A shipped game's cooked texture (see cookTextures), or null if there is none.
std::shared_ptr<const TextureMips> loadCooked(const std::filesystem::path& root, const std::string& canonical, TextureKind kind,
                                              const std::string& file, std::filesystem::path* outPath = nullptr) {
    const std::string relative = relativeTextureKey(root, canonical);
    if (relative.empty()) return nullptr;
    std::error_code ec;
    if (!std::filesystem::is_directory(root / "Cooked" / "Textures", ec)) return nullptr;
    const auto size = std::filesystem::file_size(file, ec);
    if (ec) return nullptr;
    const std::filesystem::path cooked = cookedTextureFile(root, relative, kind, size);
    if (!std::filesystem::exists(cooked, ec)) return nullptr;
    auto loaded = loadTextureMips(cooked);
    if (!loaded) {
        log::warn("Renderer", "cooked texture {}: {}", cooked.filename().string(), loaded.error().message);
        return nullptr;
    }
    if (outPath != nullptr) *outPath = cooked;
    return std::make_shared<const TextureMips>(std::move(*loaded));
}

} // namespace

TextureCache::TextureCache(RenderContext& context) : ctx_(context) {
    compression_ = ctx_.caps().textureCompressionBC;
}

TextureCache::~TextureCache() {
    for (auto& [key, entry] : textures_) {
        if (entry.preparing.valid()) entry.preparing.wait();
        if (entry.reloading.valid()) entry.reloading.wait();
        ctx_.bindless().releaseTexture(entry.bindless);
    }
    for (const Retired& r : retired_) ctx_.bindless().releaseTexture(r.bindless);
}

std::size_t TextureCache::invalidateFile(const std::string& file) {
    auto fromFile = [&](const std::string& id) {
        return id.starts_with(file) && id.size() > file.size() && (id[file.size()] == '|' || id[file.size()] == '#');
    };
    std::erase_if(failed_, fromFile);
    std::size_t dropped = 0;
    for (auto it = textures_.begin(); it != textures_.end();) {
        if (fromFile(it->first)) {
            if (it->second.preparing.valid()) it->second.preparing.wait();
            if (it->second.reloading.valid()) it->second.reloading.wait();
            ctx_.bindless().releaseTexture(it->second.bindless);  // the GPU is idle
            it = textures_.erase(it);
            ++dropped;
        } else {
            ++it;
        }
    }
    return dropped;
}

std::uint32_t TextureCache::levelFor(std::uint32_t width, std::uint32_t height, std::size_t levelCount, float wanted) {
    if (levelCount == 0) return 0;
    const float largest = static_cast<float>(std::max(width, height));
    if (!(wanted > 0.0f) || wanted >= largest) return 0;
    const auto level = static_cast<std::uint32_t>(std::floor(std::log2(largest / std::max(wanted, 1.0f))));
    return std::min(level, static_cast<std::uint32_t>(levelCount - 1));
}

std::uint32_t TextureCache::levelFor(const TextureMips& mips, float wanted) {
    return levelFor(mips.width, mips.height, mips.levels.size(), wanted);
}

std::size_t TextureCache::entryResidentBytes(const Entry& entry, std::uint32_t firstLevel) noexcept {
    std::size_t total = 0;
    for (std::size_t i = firstLevel; i < entry.levelBytes.size(); ++i) {
        total += entry.levelBytes[i];
    }
    return total;
}

void TextureCache::recordMetadata(Entry& entry, const TextureMips& mips) {
    entry.width = mips.width;
    entry.height = mips.height;
    entry.format = mips.format;
    entry.levelBytes.resize(mips.levels.size());
    for (std::size_t i = 0; i < mips.levels.size(); ++i) {
        entry.levelBytes[i] = mips.levels[i].size();
    }
}

void TextureCache::retire(Entry& entry) {
    if (entry.texture) retired_.push_back({entry.texture, entry.bindless, frame_});
    entry.texture = nullptr;
    entry.bindless = UINT32_MAX;
}

void TextureCache::uploadFromMips(Entry& entry, const TextureMips& mips, std::uint32_t firstLevel) {
    std::vector<std::span<const std::byte>> levels;
    for (std::size_t l = firstLevel; l < mips.levels.size(); ++l) levels.push_back(std::as_bytes(std::span(mips.levels[l])));
    nvrhi::TextureHandle texture =
        ctx_.createTexture(levels, mips.levelWidth(firstLevel), mips.levelHeight(firstLevel), mips.format, entry.name.c_str());
    retire(entry);
    entry.texture = texture;
    entry.bindless = ctx_.bindless().registerTexture(texture);
    entry.firstLevel = firstLevel;
    entry.pendingLevel = UINT32_MAX;
    entry.lastMipChangeFrame = frame_;
}

void TextureCache::upload(Entry& entry, std::uint32_t firstLevel) {
    uploadFromMips(entry, *entry.mips, firstLevel);
}

std::uint32_t TextureCache::index(AssetLibrary& library, const std::string& key, TextureKind kind, bool block, float wanted) {
    if (key.empty()) return UINT32_MAX;
    auto canonical = library.canonicalTextureKey(key);
    if (!canonical) return UINT32_MAX;
    const std::string id = *canonical + textureKindSuffix(kind);
    if (failed_.contains(id)) return UINT32_MAX;
    Entry& entry = textures_[id];
    entry.lastUsed = frame_;
    entry.wanted = std::max(entry.wanted, wanted);
    if (entry.texture) return entry.bindless;

    if (!entry.mips) {
        if (!entry.preparing.valid()) {
            entry.name = key;
            const std::string file = canonical->substr(0, canonical->find('#'));
            // Block-compressed data only helps a GPU that reads it.
            if (compression_) entry.mips = loadCooked(library.root(), *canonical, kind, file, &entry.diskPath);
            const std::filesystem::path cached = entry.mips ? std::filesystem::path{} : textureCacheFile(library.root(), id, file);
            if (!cached.empty() && std::filesystem::exists(cached)) {
                if (auto loaded = loadTextureMips(cached)) {
                    entry.diskPath = cached;
                    entry.mips = std::make_shared<const TextureMips>(std::move(*loaded));
                } else {
                    log::warn("Renderer", "texture cache {}: {}", cached.filename().string(), loaded.error().message);
                }
            }
            if (!entry.mips) {
                const ImageData* data = nullptr;
                if (block) {
                    auto loaded = library.texture(key);
                    if (loaded) data = *loaded;
                } else {
                    const auto request = library.requestTexture(key);
                    if (request.state == AssetState::Pending) return UINT32_MAX;  // untextured until decoded
                    data = request.value;
                }
                if (data == nullptr) {
                    textures_.erase(id);
                    failed_.insert(id);
                    return UINT32_MAX;
                }
                const bool compress = compression_;
                if (!cached.empty() && compress) entry.diskPath = cached;
                entry.preparing =
                    JobSystem::global()
                        .submit([rgba = data->rgba, width = data->width, height = data->height, kind, compress, cached,
                                 name = key]() -> std::shared_ptr<const TextureMips> {
                            Stopwatch timer;
                            auto mips = std::make_shared<TextureMips>(prepareTexture(rgba, width, height, kind, compress));
                            if (!cached.empty() && compress) {
                                if (auto saved = saveTextureMips(cached, *mips); !saved) {
                                    log::warn("Renderer", "texture cache for '{}': {}", name, saved.error().message);
                                }
                            }
                            log::debug("Renderer", "texture '{}' {}x{} prepared in {:.1f} ms", name, width, height, timer.elapsedMs());
                            return mips;
                        })
                        .share();
            }
        }
        if (!entry.mips) {
            if (!block && entry.preparing.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return UINT32_MAX;
            entry.mips = entry.preparing.get();
            entry.preparing = {};
        }
    }
    recordMetadata(entry, *entry.mips);
    std::uint32_t first = levelFor(*entry.mips, entry.wanted);
    if (!block) first = std::max(first, levelFor(*entry.mips, static_cast<float>(kFirstAppearance)));
    upload(entry, first);
    return entry.bindless;
}

void TextureCache::beginFrame() {
    ++frame_;
    // GPU textures replaced a few frames ago are no longer in use.
    std::erase_if(retired_, [&](const Retired& r) {
        if (r.frame + kRetireFrames > frame_) return false;
        ctx_.bindless().releaseTexture(r.bindless);
        return true;
    });
    // Finish any background disk reloads before planning this frame's streaming.
    for (auto& [id, entry] : textures_) {
        if (entry.reloading.valid() && entry.reloading.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            if (auto loaded = entry.reloading.get()) {
                entry.mips = std::move(loaded);
                recordMetadata(entry, *entry.mips);
            }
            entry.reloading = {};
        }
    }
    // What each texture should have: the finest level anyone asked for last frame. Finer first
    // for the most needed (largest step), within the upload budget of a frame.
    struct Want {
        Entry* entry;
        std::uint32_t level;
    };
    std::vector<Want> finer;
    std::size_t resident = 0;
    for (auto& [id, entry] : textures_) {
        if (!entry.texture || entry.levelBytes.empty()) {
            entry.wanted = 0.0f;
            continue;
        }
        resident += entryResidentBytes(entry, entry.firstLevel);
        std::uint32_t desired = entry.pendingLevel;
        if (entry.lastUsed + 1 >= frame_) {
            const std::uint32_t level = levelFor(entry.width, entry.height, entry.levelBytes.size(), entry.wanted);
            desired = std::min(desired, level);
        }
        if (desired < entry.firstLevel) {
            entry.pendingLevel = desired;
            finer.push_back({&entry, desired});
        }
        entry.wanted = 0.0f;
    }
    std::ranges::sort(finer, [](const Want& a, const Want& b) {
        return a.entry->firstLevel - a.level > b.entry->firstLevel - b.level;
    });
    streamedIn_ = 0;
    reduced_ = 0;
    std::size_t uploaded = 0;
    for (const Want& w : finer) {
        if (!w.entry->mips) {
            if (!w.entry->diskPath.empty() && !w.entry->reloading.valid()) {
                std::filesystem::path diskPath = w.entry->diskPath;
                std::string name = w.entry->name;
                w.entry->reloading =
                    JobSystem::global()
                        .submit([path = std::move(diskPath), name = std::move(name)]() -> std::shared_ptr<const TextureMips> {
                            auto loaded = loadTextureMips(path);
                            if (!loaded) {
                                log::warn("Renderer", "texture reload '{}': {}", name, loaded.error().message);
                                return nullptr;
                            }
                            return std::make_shared<const TextureMips>(std::move(*loaded));
                        })
                        .share();
            }
            continue;
        }
        const std::size_t bytes = entryResidentBytes(*w.entry, w.level);
        if (uploaded > 0 && uploaded + bytes > uploadPerFrame_) break;
        const std::size_t extra = bytes - entryResidentBytes(*w.entry, w.entry->firstLevel);
        if (resident + extra > budget_) continue;  // would only be taken away again
        resident += extra;
        upload(*w.entry, w.level);
        uploaded += bytes;
        ++streamedIn_;
    }
    if (resident > budget_) {
        // Over budget: drop what nothing used for a while (oldest first), then take the finest level
        // off the largest textures in use until it fits.
        std::vector<Entry*> unused;
        std::vector<Entry*> used;
        for (auto& [id, entry] : textures_) {
            if (!entry.texture) continue;
            (entry.lastUsed + kUnusedFrames < frame_ ? unused : used).push_back(&entry);
        }
        std::ranges::sort(unused, std::less<>{}, &Entry::lastUsed);
        bool underBudget = false;
        for (Entry* e : unused) {
            if (resident <= budget_) {
                underBudget = true;
                break;
            }
            resident -= entryResidentBytes(*e, e->firstLevel);
            retire(*e);
            ++reduced_;
        }
        if (!underBudget && resident > budget_) {
            struct Plan {
                Entry* entry;
                std::uint32_t level;
            };
            std::vector<Plan> plans;
            plans.reserve(used.size());
            for (Entry* e : used) plans.push_back({e, e->firstLevel});
            for (bool progress = true; resident > budget_ && progress;) {
                progress = false;
                std::ranges::sort(plans, [](const Plan& a, const Plan& b) {
                    return entryResidentBytes(*a.entry, a.level) > entryResidentBytes(*b.entry, b.level);
                });
                for (Plan& p : plans) {
                    if (resident <= budget_) break;
                    if (p.level + 1 >= p.entry->levelBytes.size()) continue;
                    resident -= entryResidentBytes(*p.entry, p.level) - entryResidentBytes(*p.entry, p.level + 1);
                    ++p.level;
                    progress = true;
                }
            }
            for (const Plan& p : plans) {
                if (p.level == p.entry->firstLevel) continue;
                if (p.entry->mips) {
                    upload(*p.entry, p.level);
                    ++reduced_;
                } else if (!p.entry->diskPath.empty()) {
                    if (auto loaded = loadTextureMips(p.entry->diskPath)) {
                        uploadFromMips(*p.entry, *loaded, p.level);
                        ++reduced_;
                    }
                }
            }
        }
    }

    // Release CPU mip chains for textures backed by disk cache once their resident mips settle.
    for (auto& [id, entry] : textures_) {
        if (!entry.texture || !entry.mips || entry.diskPath.empty() || entry.reloading.valid()) continue;
        if (entry.pendingLevel < entry.firstLevel) continue;
        if (frame_ >= entry.lastMipChangeFrame + ramReleaseIdleFrames_) {
            std::error_code ec;
            if (std::filesystem::exists(entry.diskPath, ec)) {
                entry.mips.reset();
            }
        }
    }
}

TextureCache::Stats TextureCache::stats() const {
    Stats s;
    for (const auto& [id, entry] : textures_) {
        if (entry.mips) s.cpuCacheBytes += entry.mips->bytes(0);
        if (!entry.texture || entry.levelBytes.empty()) continue;
        ++s.textures;
        s.residentBytes += entryResidentBytes(entry, entry.firstLevel);
        if (entry.format != nvrhi::Format::RGBA8_UNORM && entry.format != nvrhi::Format::SRGBA8_UNORM) ++s.compressed;
    }
    s.streamedIn = streamedIn_;
    s.reduced = reduced_;
    return s;
}

} // namespace lumora::render
