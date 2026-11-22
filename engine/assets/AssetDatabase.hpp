#pragma once

#include "core/Error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace lumora {

enum class AssetKind : std::uint8_t {
    Model,
    Texture,
    Material,
    Scene,
    Audio,
    Script,
    Other,
};

[[nodiscard]] std::string_view toString(AssetKind kind) noexcept;
[[nodiscard]] AssetKind classifyAssetKind(const std::filesystem::path& path);

struct AssetRecord {
    std::string key;
    AssetKind kind = AssetKind::Other;
    std::uint64_t sizeBytes = 0;
    std::int64_t mtimeNs = 0;
    std::uint64_t contentHash = 0;  // 64-bit FNV-1a of source file bytes
    std::string cookedPath;
    bool cookedValid = false;

    friend bool operator==(const AssetRecord&, const AssetRecord&) = default;
};

struct MeshCookStats {
    std::size_t cooked = 0;
    std::size_t skippedUpToDate = 0;
    std::size_t failed = 0;
    std::size_t bytesWritten = 0;
};

/// Tracks project content files by relative key, size, mtime, FNV-1a content hash, and cook status.
class AssetDatabase {
public:
    /// Recursively scans `projectRoot` (skipping Saved/, Cooked/, .git/) and updates records.
    /// Existing unchanged records preserve their `cookedPath` and `cookedValid` state.
    [[nodiscard]] Result<std::size_t> scanContent(const std::filesystem::path& projectRoot);

    /// Persists the database as JSON using `FileSystem.hpp` atomic text writes.
    [[nodiscard]] Result<> save(const std::filesystem::path& path) const;

    /// Loads a previously saved JSON asset database.
    [[nodiscard]] Result<> load(const std::filesystem::path& path);

    /// Returns sorted keys of files on disk that are new, modified (size/hash), or uncooked models.
    [[nodiscard]] std::vector<std::string> findStaleAssets(const std::filesystem::path& projectRoot) const;

    /// Discovers and cooks any new, modified, or missing `.lmesh` outputs for Model assets.
    [[nodiscard]] Result<MeshCookStats> cookStaleMeshes(const std::filesystem::path& projectRoot,
                                                        const std::filesystem::path& outputRoot);

    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }
    [[nodiscard]] const AssetRecord* find(std::string_view key) const noexcept;
    [[nodiscard]] std::vector<AssetRecord> records() const;
    void clear() noexcept { records_.clear(); }

private:
    std::map<std::string, AssetRecord, std::less<>> records_;
};

} // namespace lumora
