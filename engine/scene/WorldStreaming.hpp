#pragma once

#include "core/Error.hpp"
#include "core/Uuid.hpp"

#include <entt/entity/entity.hpp>
#include <glm/vec3.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lumora {

class Scene;

/// Discrete 2D sector coordinate on the world XZ streaming grid.
struct SectorCoord {
    std::int32_t x = 0;
    std::int32_t z = 0;

    friend constexpr bool operator==(const SectorCoord&, const SectorCoord&) noexcept = default;
};

/// Mixes 32-bit (x, z) grid coordinates into a 64-bit hash with SplitMix64 avalanche.
struct SectorCoordHash {
    [[nodiscard]] std::size_t operator()(const SectorCoord& coord) const noexcept {
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(coord.x)) << 32) |
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(coord.z));
        std::uint64_t h = key + 0x9e3779b97f4a7c15ull;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
        h = h ^ (h >> 31);
        return static_cast<std::size_t>(h);
    }
};

} // namespace lumora

template <>
struct std::hash<lumora::SectorCoord> {
    std::size_t operator()(const lumora::SectorCoord& coord) const noexcept {
        return lumora::SectorCoordHash{}(coord);
    }
};

namespace lumora {

/// Hierarchical level of detail for spatial streaming sectors (0 = finest, 2 = coarsest).
enum class HlodLevel : std::uint8_t {
    Lod0Full = 0,
    Lod1Medium = 1,
    Lod2Coarse = 2,
    Unloaded = 3,
};

[[nodiscard]] std::string_view toString(HlodLevel level) noexcept;

/// Spatial streaming grid and HLOD hysteresis parameters.
struct StreamingConfig {
    float cellSize = 128.0f;
    float lod0Radius = 150.0f;
    float lod1Radius = 350.0f;
    float lod2Radius = 700.0f;
    float hysteresis = 16.0f;
    float commitBudgetMs = 2.0f;
};

/// Maps a world-space position onto its containing sector grid cell (XZ plane).
[[nodiscard]] SectorCoord worldToSectorCoord(const glm::vec3& worldPos, float cellSize) noexcept;

/// Euclidean distance in XZ from `focusPos` to the axis-aligned footprint of `coord`.
/// Returns 0 when `focusPos` lies inside or on the border of `coord`.
[[nodiscard]] float distanceToSectorXZ(const glm::vec3& focusPos, SectorCoord coord, float cellSize) noexcept;

/// Computes the target HLOD level at `distance` using dual-threshold Schmitt-trigger
/// hysteresis relative to `currentLod` to prevent border thrashing.
[[nodiscard]] HlodLevel evaluateSectorHlod(float distance, HlodLevel currentLod, const StreamingConfig& config) noexcept;

/// Returns `<sectorRoot>/sector_<x>_<z>.lsector`.
[[nodiscard]] std::filesystem::path sectorFilePath(const std::filesystem::path& sectorRoot, SectorCoord coord);

/// Serialized entity entry inside a `.lsector` package.
/// `minLod` specifies the coarsest HLOD level at which this entity remains visible
/// (e.g. `Lod0Full` = full-detail only; `Lod2Coarse` = visible across Lod0, Lod1, and Lod2).
struct SectorEntityRecord {
    Uuid id;
    std::string name;
    glm::vec3 position{0.0f};
    glm::vec3 rotationDeg{0.0f};
    glm::vec3 scale{1.0f};
    std::string meshAsset;
    std::string materialAsset;
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    HlodLevel minLod = HlodLevel::Lod2Coarse;

    friend bool operator==(const SectorEntityRecord&, const SectorEntityRecord&) = default;
};

/// Binary `.lsector` package describing all streamed entities for one sector cell.
struct SectorPackage {
    SectorCoord coord{};
    glm::vec3 boundsMin{0.0f};
    glm::vec3 boundsMax{0.0f};
    std::vector<SectorEntityRecord> entities;

    friend bool operator==(const SectorPackage&, const SectorPackage&) = default;
};

/// 64-bit FNV-1a checksum over a raw byte span.
[[nodiscard]] std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept;

/// Deterministic LZ4-style block compressor (zero external dependencies).
[[nodiscard]] std::vector<std::byte> compressSectorBlock(std::span<const std::byte> input);

/// Deterministic LZ4-style block decompressor with strict bounds validation.
[[nodiscard]] Result<std::vector<std::byte>> decompressSectorBlock(std::span<const std::byte> compressed,
                                                                   std::size_t expectedSize);

/// Encodes `pkg` into `.lsector` binary container bytes (header + compressed or raw payload).
[[nodiscard]] Result<std::vector<std::byte>> encodeSectorPackage(const SectorPackage& pkg);

/// Validates magic, version, bounds, and FNV-1a checksum before decoding `SectorPackage`.
[[nodiscard]] Result<SectorPackage> decodeSectorPackage(std::span<const std::byte> bytes);

/// Atomically writes a `.lsector` package file to disk.
[[nodiscard]] Result<> saveSectorPackage(const std::filesystem::path& file, const SectorPackage& pkg);

/// Reads and validates a `.lsector` package file from disk.
[[nodiscard]] Result<SectorPackage> loadSectorPackage(const std::filesystem::path& file);

/// Component attached to every entity spawned by `WorldStreamer` for sector ownership tracking.
struct StreamedSectorTag {
    SectorCoord coord{};
    HlodLevel minLod = HlodLevel::Lod2Coarse;
};

/// Decompressed sector payload built on the background I/O thread and awaiting main-thread commit.
struct StagedSector {
    SectorCoord coord{};
    HlodLevel targetLod = HlodLevel::Lod0Full;
    SectorPackage package;
};

/// Runtime telemetry snapshot for `WorldStreamer`.
struct StreamingStats {
    std::size_t activeSectors = 0;
    std::size_t loadingSectors = 0;
    std::size_t stagedSectors = 0;
    std::size_t activeEntities = 0;
    std::uint64_t loadsCompleted = 0;
    std::uint64_t unloadsCompleted = 0;
};

/// Asynchronous open-world spatial streamer with background disk I/O + decompression
/// and time-budgeted main-thread ECS commit.
class WorldStreamer {
public:
    explicit WorldStreamer(std::filesystem::path sectorRoot, StreamingConfig config = {});
    ~WorldStreamer();

    WorldStreamer(const WorldStreamer&) = delete;
    WorldStreamer& operator=(const WorldStreamer&) = delete;
    WorldStreamer(WorldStreamer&&) = delete;
    WorldStreamer& operator=(WorldStreamer&&) = delete;

    /// Evaluates the spatial grid around `focusPos` with hysteresis, enqueueing missing sectors
    /// for background loading and marking out-of-range sectors for unload.
    void updateFocus(const glm::vec3& focusPos);

    /// Commits staged sectors and destroys unloaded sector entities in `scene`,
    /// respecting `config().commitBudgetMs` (non-positive disables time slicing).
    void commitReadySectors(Scene& scene);

    /// Thread-safe snapshot of streaming counters.
    [[nodiscard]] StreamingStats stats() const;

    /// Returns the current effective HLOD state of `coord` (`Unloaded` if not loaded/staged/loading).
    [[nodiscard]] HlodLevel sectorHlod(SectorCoord coord) const;

    /// Returns live entity handles currently spawned in `Scene` for `coord`.
    [[nodiscard]] std::vector<entt::entity> entitiesInSector(SectorCoord coord) const;

    /// Clears cached missing/corrupt state for `coord` so a subsequent `updateFocus` retries loading it.
    void invalidateSector(SectorCoord coord);

    /// Stops and joins the background I/O thread (idempotent; also called on destruction).
    void shutdown();

    [[nodiscard]] const std::filesystem::path& sectorRoot() const noexcept { return sectorRoot_; }
    [[nodiscard]] const StreamingConfig& config() const noexcept { return config_; }
    void setConfig(const StreamingConfig& config);

private:
    struct LoadRequest {
        SectorCoord coord{};
        HlodLevel targetLod = HlodLevel::Lod0Full;
        std::uint64_t generation = 0;
        float distSq = 0.0f;
    };

    struct PendingState {
        std::uint64_t generation = 0;
        HlodLevel targetLod = HlodLevel::Lod0Full;
    };

    enum class DirtyState : std::uint8_t { None = 0, LodUpdate = 1, Unload = 2 };

    struct ActiveSector {
        SectorCoord coord{};
        HlodLevel activeLod = HlodLevel::Unloaded;
        HlodLevel desiredLod = HlodLevel::Unloaded;
        SectorPackage package;
        std::vector<entt::entity> entities;
        std::size_t reconcileCursor = 0;
        DirtyState dirtyState = DirtyState::None;
        bool needsReconcile = false;
    };

    void workerLoop(std::stop_token stopToken);
    void scheduleDirtyLockedMain(SectorCoord coord, ActiveSector& sector);
    bool reconcileSectorStep(Scene& scene, ActiveSector& sector, bool useBudget,
                             std::chrono::steady_clock::time_point startTime,
                             std::chrono::duration<double, std::milli> budget, std::size_t& opsDone);

    std::filesystem::path sectorRoot_;
    StreamingConfig config_;

    // Main-thread sector tracking (also locked for const cross-thread inspection).
    mutable std::mutex mainStateMutex_;
    std::unordered_map<SectorCoord, ActiveSector> activeSectors_;
    std::optional<ActiveSector> partialCommit_;
    std::deque<SectorCoord> dirtyUnloads_;
    std::deque<SectorCoord> dirtyLodUpdates_;

    // Shared queue state between main thread and background I/O thread.
    mutable std::mutex queueMutex_;
    std::condition_variable_any queueCv_;
    std::vector<LoadRequest> loadQueue_;
    std::unordered_map<SectorCoord, PendingState> pendingMap_;
    std::deque<StagedSector> stagedQueue_;
    std::unordered_set<SectorCoord> absentSectors_;
    std::uint64_t nextGeneration_ = 0;

    std::atomic<std::size_t> activeSectorsCount_{0};
    std::atomic<std::size_t> activeEntitiesCount_{0};
    std::atomic<std::uint64_t> loadsCompleted_{0};
    std::atomic<std::uint64_t> unloadsCompleted_{0};
    std::atomic<bool> partialCommitActive_{false};
    std::atomic<bool> stopped_{false};

    // Declared last so it joins before any state it might access during shutdown.
    std::jthread ioThread_;
};

} // namespace lumora
