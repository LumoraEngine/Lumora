#include "renderer/TextureCooker.hpp"

#include "assets/AssetImport.hpp"
#include "assets/AssetLibrary.hpp"
#include "assets/Image.hpp"
#include "assets/MaterialGraph.hpp"
#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "core/Timer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <format>
#include <optional>
#include <set>

namespace lumora::render {

namespace {

namespace stdfs = std::filesystem;

enum class FieldUse { None, Color, Data, Normal, Other };

/// What a JSON field holding an asset key uses it for.
FieldUse fieldUse(std::string_view name) {
    if (name == "normalTexture") return FieldUse::Normal;
    if (name == "metallicRoughnessTexture" || name == "occlusionTexture") return FieldUse::Data;
    if (name == "baseColorTexture" || name == "emissiveTexture" || name == "decalTexture" || name == "texture" || name == "image") {
        return FieldUse::Color;
    }
    if (name == "heightmap") return FieldUse::Other;
    return FieldUse::None;
}

bool looksLikeTexture(std::string_view value) {
    if (value.empty()) return false;
    if (const auto hash = value.rfind('#'); hash != std::string_view::npos) return value.substr(hash + 1).starts_with("tex");
    return isImageFile(stdfs::path(std::string(value)));
}

bool searched(const stdfs::path& relative) {
    const std::string first = relative.begin()->string();
    return first != "Saved" && first != "Cooked" && first != ".git";
}

std::string lowerExtension(const stdfs::path& file) {
    std::string ext = file.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

class Collector {
public:
    explicit Collector(const stdfs::path& root) : root_(root), library_(root) {}

    /// A key as found in a file: kept in its project-relative canonical form.
    void use(std::string_view key, std::optional<TextureKind> kind) {
        auto canonical = library_.canonicalTextureKey(key);
        if (!canonical) return;
        std::string relative = relativeTextureKey(root_, *canonical);
        if (relative.empty()) return;
        named_.insert(relative);
        if (kind) uses_.insert({std::move(relative), *kind});
    }

    void walk(const nlohmann::json& j, std::string_view field) {
        if (j.is_object()) {
            for (const auto& [name, value] : j.items()) walk(value, name);
        } else if (j.is_array()) {
            for (const auto& value : j) walk(value, field);
        } else if (j.is_string()) {
            const FieldUse use = fieldUse(field);
            const std::string& text = j.get_ref<const std::string&>();
            if (use == FieldUse::None || !looksLikeTexture(text)) return;
            switch (use) {
                case FieldUse::Color: this->use(text, TextureKind::Color); break;
                case FieldUse::Data: this->use(text, TextureKind::Data); break;
                case FieldUse::Normal: this->use(text, TextureKind::Normal); break;
                default: this->use(text, std::nullopt); break;
            }
        }
    }

    void file(const stdfs::path& absolute, const stdfs::path& relative) {
        const std::string ext = lowerExtension(absolute);
        if (ext == ".lscene" || ext == ".lprefab" || ext == ".lmat" || ext == ".lui" || ext == ".lgraph") {
            auto text = fs::readTextFile(absolute);
            if (!text) return;
            const nlohmann::json j = nlohmann::json::parse(*text, nullptr, false);
            if (!j.is_discarded()) walk(j, {});
        } else if (ext == kMaterialGraphExtension) {
            auto program = loadMaterialGraph(absolute);
            if (!program) return;
            for (const MaterialProgram::Texture& t : program->textures) {
                use(t.key, t.kind == matgraph::TextureKind::Normal ? TextureKind::Normal
                           : t.kind == matgraph::TextureKind::Data ? TextureKind::Data
                                                                    : TextureKind::Color);
            }
        } else if (isModelFile(absolute)) {
            auto model = library_.model(absolute);
            if (!model) return;
            const std::string file = relative.generic_string();
            auto map = [&](int image, TextureKind kind) {
                if (image >= 0) use(AssetLibrary::textureKey(file, image), kind);
            };
            for (const ImportedMaterial& m : (*model)->materials) {
                map(m.baseColorTexture, TextureKind::Color);
                map(m.emissiveTexture, TextureKind::Color);
                map(m.metallicRoughnessTexture, TextureKind::Data);
                map(m.occlusionTexture, TextureKind::Data);
                map(m.normalTexture, TextureKind::Normal);
            }
            library_.invalidateFile(absolute);  // the cooking decodes one model at a time
        } else if (isImageFile(absolute)) {
            images_.push_back(relative.generic_string());
        }
    }

    std::vector<TextureUse> finish() {
        for (const std::string& image : images_) {
            if (!named_.contains(image)) uses_.insert({image, TextureKind::Color});
        }
        std::vector<TextureUse> out;
        out.reserve(uses_.size());
        for (const auto& [key, kind] : uses_) out.push_back({key, kind});
        return out;
    }

private:
    stdfs::path root_;
    AssetLibrary library_;
    std::set<std::pair<std::string, TextureKind>> uses_;
    std::set<std::string> named_;
    std::vector<std::string> images_;
};

} // namespace

std::vector<TextureUse> collectTextureUses(const stdfs::path& projectRoot) {
    Collector collector(projectRoot);
    std::error_code ec;
    for (auto it = stdfs::recursive_directory_iterator(projectRoot, ec); !ec && it != stdfs::recursive_directory_iterator();
         it.increment(ec)) {
        const stdfs::path relative = it->path().lexically_relative(projectRoot);
        if (!searched(relative)) {
            if (it->is_directory(ec)) it.disable_recursion_pending();
            continue;
        }
        if (it->is_regular_file(ec)) collector.file(it->path(), relative);
    }
    return collector.finish();
}

Result<CookStats> cookTextures(const TextureCooking& request) {
    Stopwatch timer;
    const std::vector<TextureUse> uses = collectTextureUses(request.projectRoot);
    AssetLibrary library(request.projectRoot);
    CookStats stats;
    auto warn = [&](std::string message) {
        log::warn("Cook", "{}", message);
        if (request.warnings != nullptr) request.warnings->push_back(std::move(message));
        ++stats.failed;
    };
    std::string loadedFile;  // the source decoded last: dropped when the next one starts
    for (std::size_t i = 0; i < uses.size(); ++i) {
        const TextureUse& use = uses[i];
        if (request.progress && !request.progress(i, uses.size(), use.key)) return makeError("cooking cancelled");
        auto canonical = library.canonicalTextureKey(use.key);
        if (!canonical) {
            warn(std::format("texture {}: {}", use.key, canonical.error().message));
            continue;
        }
        const std::string file = canonical->substr(0, canonical->find('#'));
        if (file != loadedFile) {
            if (!loadedFile.empty()) library.invalidateFile(loadedFile);
            loadedFile = file;
        }
        std::error_code ec;
        const auto size = stdfs::file_size(file, ec);
        if (ec) {
            warn(std::format("texture {}: {}", use.key, ec.message()));
            continue;
        }
        const stdfs::path out = cookedTextureFile(request.outputRoot, use.key, use.kind, size);
        if (stdfs::exists(out, ec)) continue;
        if (!stdfs::create_directories(out.parent_path(), ec) && ec) {
            return makeError("cannot create {}: {}", out.parent_path().string(), ec.message());
        }

        // The editor compressed it already: same preparation, same bytes.
        const stdfs::path cached = textureCacheFile(request.projectRoot, *canonical + textureKindSuffix(use.kind), file);
        if (!cached.empty() && loadTextureMips(cached).has_value() && stdfs::copy_file(cached, out, ec)) {
            ++stats.reused;
            stats.bytes += stdfs::file_size(out, ec);
            continue;
        }
        auto image = library.texture(use.key);
        if (!image) {
            warn(std::format("texture {}: {}", use.key, image.error().message));
            continue;
        }
        const TextureMips mips = prepareTexture((*image)->rgba, (*image)->width, (*image)->height, use.kind, true);
        if (auto saved = saveTextureMips(out, mips); !saved) return makeError("{}: {}", out.string(), saved.error().message);
        ++stats.cooked;
        stats.bytes += stdfs::file_size(out, ec);
    }
    if (request.progress) request.progress(uses.size(), uses.size(), {});
    log::info("Cook", "{} textures cooked, {} from the editor's cache, {} failed ({:.1f} MB) in {:.1f} s", stats.cooked, stats.reused,
              stats.failed, static_cast<double>(stats.bytes) / (1024.0 * 1024.0), timer.elapsedMs() / 1000.0);
    return stats;
}

} // namespace lumora::render
