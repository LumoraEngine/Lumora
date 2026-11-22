#include "assets/AssetDatabase.hpp"

#include "assets/AssetImport.hpp"
#include "assets/GltfImporter.hpp"
#include "assets/Image.hpp"
#include "assets/Material.hpp"
#include "assets/MeshCooker.hpp"
#include "core/FileSystem.hpp"
#include "core/Log.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <format>
#include <fstream>
#include <optional>
#include <system_error>
#include <unordered_set>

namespace lumora {

namespace {

namespace stdfs = std::filesystem;

constexpr int kAssetDbFormatVersion = 1;

std::string lowerExtension(const stdfs::path& path) {
    std::string ext = path.extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

bool shouldSkipRelative(const stdfs::path& relative) {
    if (relative.empty()) return false;
    const std::string first = relative.begin()->string();
    if (first == "Saved" || first == "Cooked" || first == ".git") return true;
    for (const auto& part : relative) {
        if (part.string() == ".git") return true;
    }
    return false;
}

std::uint64_t fnv1aString(std::string_view text) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char c : text) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

Result<std::uint64_t> hashFileFnv1a(const stdfs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        return makeError("cannot open '{}' for hashing", file.string());
    }
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    std::array<char, 65536> buffer{};
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = in.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]));
            hash *= 0x100000001b3ULL;
        }
    }
    if (in.bad()) {
        return makeError("failed reading '{}' while hashing", file.string());
    }
    return hash;
}

Result<std::int64_t> readMtimeNs(const stdfs::path& file) {
    std::error_code ec;
    const auto ftime = stdfs::last_write_time(file, ec);
    if (ec) {
        return makeError("cannot read modification time for '{}': {}", file.string(), ec.message());
    }
    // Convert Windows FILETIME (1601 epoch) to Unix epoch (1970) before scaling to ns so int64_t never overflows.
#ifdef _WIN32
    constexpr std::chrono::seconds kWindowsToUnixEpoch{11644473600LL};
    const auto adjusted = ftime.time_since_epoch() - kWindowsToUnixEpoch;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(adjusted).count();
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(ftime.time_since_epoch()).count();
#endif
}

Result<AssetRecord> inspectFile(const stdfs::path& file, std::string key) {
    std::error_code ec;
    const auto size = stdfs::file_size(file, ec);
    if (ec) {
        return makeError("cannot read size of '{}': {}", file.string(), ec.message());
    }
    auto mtime = readMtimeNs(file);
    if (!mtime) return std::unexpected(mtime.error());

    auto hash = hashFileFnv1a(file);
    if (!hash) return std::unexpected(hash.error());

    AssetRecord rec;
    rec.key = std::move(key);
    rec.kind = classifyAssetKind(file);
    rec.sizeBytes = static_cast<std::uint64_t>(size);
    rec.mtimeNs = *mtime;
    rec.contentHash = *hash;
    rec.cookedValid = false;
    return rec;
}

std::optional<AssetKind> parseAssetKind(std::string_view name) noexcept {
    if (name == "Model") return AssetKind::Model;
    if (name == "Texture") return AssetKind::Texture;
    if (name == "Material") return AssetKind::Material;
    if (name == "Scene") return AssetKind::Scene;
    if (name == "Audio") return AssetKind::Audio;
    if (name == "Script") return AssetKind::Script;
    if (name == "Other") return AssetKind::Other;
    return std::nullopt;
}

std::string cookedMeshRelativePath(std::string_view key, std::size_t meshIndex) {
    return std::format("Cooked/Meshes/{:016x}_{}.lmesh", fnv1aString(key), meshIndex);
}

template <class Callback>
Result<> visitProjectFiles(const stdfs::path& projectRoot, Callback&& callback) {
    std::error_code ec;
    if (!stdfs::is_directory(projectRoot, ec)) {
        return makeError("project root '{}' is not a directory", projectRoot.string());
    }
    for (auto it = stdfs::recursive_directory_iterator(projectRoot, ec);
         !ec && it != stdfs::recursive_directory_iterator(); it.increment(ec)) {
        const stdfs::path relative = it->path().lexically_relative(projectRoot);
        if (shouldSkipRelative(relative)) {
            if (it->is_directory(ec)) it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        if (lowerExtension(it->path()) == ".tmp") continue;
        callback(it->path(), relative.generic_string());
    }
    if (ec) {
        return makeError("failed scanning '{}': {}", projectRoot.string(), ec.message());
    }
    return {};
}

} // namespace

std::string_view toString(AssetKind kind) noexcept {
    switch (kind) {
        case AssetKind::Model: return "Model";
        case AssetKind::Texture: return "Texture";
        case AssetKind::Material: return "Material";
        case AssetKind::Scene: return "Scene";
        case AssetKind::Audio: return "Audio";
        case AssetKind::Script: return "Script";
        case AssetKind::Other: return "Other";
    }
    return "Other";
}

AssetKind classifyAssetKind(const std::filesystem::path& path) {
    if (isModelFile(path)) return AssetKind::Model;
    if (isImageFile(path) || isHdrFile(path)) return AssetKind::Texture;

    const std::string ext = lowerExtension(path);
    if (ext == ".lmat" || ext == ".lmgraph" || isMaterialFile(path)) return AssetKind::Material;
    if (ext == ".lscene" || ext == ".lprefab") return AssetKind::Scene;
    if (ext == ".wav" || ext == ".flac" || ext == ".mp3" || ext == ".ogg") return AssetKind::Audio;
    if (ext == ".lua" || ext == ".lgraph") return AssetKind::Script;
    return AssetKind::Other;
}

Result<std::size_t> AssetDatabase::scanContent(const std::filesystem::path& projectRoot) {
    std::map<std::string, AssetRecord, std::less<>> updated;
    auto walkResult = visitProjectFiles(projectRoot, [&](const stdfs::path& absolute, std::string key) {
        auto inspected = inspectFile(absolute, key);
        if (!inspected) {
            log::warn("AssetDatabase", "{}", inspected.error().message);
            return;
        }
        if (const auto existing = records_.find(key); existing != records_.end()) {
            const AssetRecord& prev = existing->second;
            if (prev.sizeBytes == inspected->sizeBytes && prev.contentHash == inspected->contentHash) {
                inspected->cookedPath = prev.cookedPath;
                inspected->cookedValid = prev.cookedValid;
            }
        }
        updated.emplace(std::move(key), std::move(*inspected));
    });
    if (!walkResult) return std::unexpected(walkResult.error());
    records_ = std::move(updated);
    return records_.size();
}

Result<> AssetDatabase::save(const std::filesystem::path& path) const {
    nlohmann::json assets = nlohmann::json::array();
    for (const auto& [key, rec] : records_) {
        assets.push_back({
            {"key", rec.key},
            {"kind", std::string(toString(rec.kind))},
            {"sizeBytes", rec.sizeBytes},
            {"mtimeNs", rec.mtimeNs},
            {"contentHash", rec.contentHash},
            {"cookedPath", rec.cookedPath},
            {"cookedValid", rec.cookedValid},
        });
    }
    const nlohmann::json root = {
        {"version", kAssetDbFormatVersion},
        {"assets", std::move(assets)},
    };
    return fs::writeTextFileAtomic(path, root.dump(2));
}

Result<> AssetDatabase::load(const std::filesystem::path& path) {
    auto text = fs::readTextFile(path);
    if (!text) return std::unexpected(text.error());

    const nlohmann::json root = nlohmann::json::parse(*text, nullptr, false);
    if (root.is_discarded() || !root.is_object()) {
        return makeError("{}: invalid JSON in asset database", path.string());
    }
    if (!root.contains("version") || !root["version"].is_number_integer() ||
        root["version"].get<int>() != kAssetDbFormatVersion) {
        return makeError("{}: unsupported asset database version", path.string());
    }
    if (!root.contains("assets") || !root["assets"].is_array()) {
        return makeError("{}: missing 'assets' array in asset database", path.string());
    }

    std::map<std::string, AssetRecord, std::less<>> loaded;
    for (const auto& item : root["assets"]) {
        if (!item.is_object()) {
            return makeError("{}: malformed asset record entry", path.string());
        }
        if (!item.contains("key") || !item["key"].is_string() ||
            !item.contains("kind") || !item["kind"].is_string() ||
            !item.contains("sizeBytes") || !item["sizeBytes"].is_number_unsigned() ||
            !item.contains("mtimeNs") || !item["mtimeNs"].is_number_integer() ||
            !item.contains("contentHash") || !item["contentHash"].is_number_unsigned() ||
            !item.contains("cookedPath") || !item["cookedPath"].is_string() ||
            !item.contains("cookedValid") || !item["cookedValid"].is_boolean()) {
            return makeError("{}: missing or mistyped field in asset record", path.string());
        }

        const auto kind = parseAssetKind(item["kind"].get_ref<const std::string&>());
        if (!kind) {
            return makeError("{}: unknown asset kind '{}'", path.string(), item["kind"].get<std::string>());
        }

        AssetRecord rec;
        rec.key = item["key"].get<std::string>();
        rec.kind = *kind;
        rec.sizeBytes = item["sizeBytes"].get<std::uint64_t>();
        rec.mtimeNs = item["mtimeNs"].get<std::int64_t>();
        rec.contentHash = item["contentHash"].get<std::uint64_t>();
        rec.cookedPath = item["cookedPath"].get<std::string>();
        rec.cookedValid = item["cookedValid"].get<bool>();
        loaded.insert_or_assign(rec.key, std::move(rec));
    }

    records_ = std::move(loaded);
    return {};
}

std::vector<std::string> AssetDatabase::findStaleAssets(const std::filesystem::path& projectRoot) const {
    std::vector<std::string> stale;
    (void)visitProjectFiles(projectRoot, [&](const stdfs::path& absolute, const std::string& key) {
        const auto it = records_.find(key);
        if (it == records_.end()) {
            stale.push_back(key);
            return;
        }
        const AssetRecord& rec = it->second;
        auto inspected = inspectFile(absolute, key);
        if (!inspected) {
            stale.push_back(key);
            return;
        }
        if (inspected->sizeBytes != rec.sizeBytes || inspected->contentHash != rec.contentHash) {
            stale.push_back(key);
            return;
        }
        if (rec.kind == AssetKind::Model && !rec.cookedValid) {
            stale.push_back(key);
        }
    });
    std::ranges::sort(stale);
    return stale;
}

Result<MeshCookStats> AssetDatabase::cookStaleMeshes(const std::filesystem::path& projectRoot,
                                                     const std::filesystem::path& outputRoot) {
    std::unordered_set<std::string> seenModels;
    auto walkResult = visitProjectFiles(projectRoot, [&](const stdfs::path& absolute, std::string key) {
        if (classifyAssetKind(absolute) != AssetKind::Model) return;
        seenModels.insert(key);
        auto inspected = inspectFile(absolute, key);
        if (!inspected) {
            log::warn("AssetDatabase", "{}", inspected.error().message);
            return;
        }
        auto it = records_.find(key);
        if (it == records_.end()) {
            records_.emplace(std::move(key), std::move(*inspected));
        } else {
            AssetRecord& rec = it->second;
            if (rec.sizeBytes != inspected->sizeBytes || rec.contentHash != inspected->contentHash) {
                rec.sizeBytes = inspected->sizeBytes;
                rec.mtimeNs = inspected->mtimeNs;
                rec.contentHash = inspected->contentHash;
                rec.cookedPath.clear();
                rec.cookedValid = false;
            } else {
                rec.kind = AssetKind::Model;
                rec.mtimeNs = inspected->mtimeNs;
            }
        }
    });
    if (!walkResult) return std::unexpected(walkResult.error());

    // Prune model records that no longer exist on disk while leaving other asset records intact.
    for (auto it = records_.begin(); it != records_.end();) {
        if (it->second.kind == AssetKind::Model && !seenModels.contains(it->first)) {
            it = records_.erase(it);
        } else {
            ++it;
        }
    }

    MeshCookStats stats;
    for (auto& [key, rec] : records_) {
        if (rec.kind != AssetKind::Model) continue;

        std::error_code ec;
        const bool outputPresent =
            rec.cookedPath.empty() || stdfs::is_regular_file(outputRoot / stdfs::path(rec.cookedPath), ec);
        if (rec.cookedValid && outputPresent) {
            ++stats.skippedUpToDate;
            continue;
        }

        const stdfs::path sourcePath = projectRoot / stdfs::path(key);
        auto imported = importModel(sourcePath);
        if (!imported) {
            log::warn("AssetDatabase", "failed importing '{}': {}", key, imported.error().message);
            rec.cookedPath.clear();
            rec.cookedValid = false;
            ++stats.failed;
            continue;
        }

        if (imported->meshes.empty()) {
            rec.cookedPath.clear();
            rec.cookedValid = true;
            ++stats.skippedUpToDate;
            continue;
        }

        bool allOk = true;
        for (std::size_t i = 0; i < imported->meshes.size(); ++i) {
            const CookedMesh cooked = cookMesh(imported->meshes[i].data);
            const std::string relOut = cookedMeshRelativePath(key, i);
            const stdfs::path absOut = outputRoot / stdfs::path(relOut);
            if (auto written = writeCookedMesh(absOut, cooked); !written) {
                log::warn("AssetDatabase", "failed writing cooked mesh '{}': {}", absOut.string(),
                          written.error().message);
                allOk = false;
                break;
            }
            const auto fileBytes = stdfs::file_size(absOut, ec);
            if (!ec) {
                stats.bytesWritten += static_cast<std::size_t>(fileBytes);
            }
        }

        if (!allOk) {
            rec.cookedPath.clear();
            rec.cookedValid = false;
            ++stats.failed;
            continue;
        }

        rec.cookedPath = cookedMeshRelativePath(key, 0);
        rec.cookedValid = true;
        ++stats.cooked;
    }

    return stats;
}

const AssetRecord* AssetDatabase::find(std::string_view key) const noexcept {
    const auto it = records_.find(key);
    return it == records_.end() ? nullptr : &it->second;
}

std::vector<AssetRecord> AssetDatabase::records() const {
    std::vector<AssetRecord> out;
    out.reserve(records_.size());
    for (const auto& [key, rec] : records_) {
        out.push_back(rec);
    }
    return out;
}

} // namespace lumora
