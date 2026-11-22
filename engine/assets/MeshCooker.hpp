#pragma once

#include "assets/MeshData.hpp"
#include "assets/MeshLod.hpp"
#include "core/Error.hpp"

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace lumora {

/// Hardware mesh-shader cluster limits used when cooking meshlets.
inline constexpr std::size_t kMaxCookedMeshletVertices = 64;
inline constexpr std::size_t kMaxCookedMeshletTriangles = 124;

/// A local triangle cluster with precomputed bounding sphere and normal cone for culling.
struct CookedMeshlet {
    std::uint32_t vertexOffset = 0;
    std::uint32_t triangleOffset = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
    glm::vec4 boundingSphere{0.0f};  // xyz = center, w = radius
    glm::vec3 coneApex{0.0f};
    glm::vec3 coneAxis{0.0f};
    float coneCutoff = 0.0f;

    friend bool operator==(const CookedMeshlet&, const CookedMeshlet&) = default;
};

/// Offline-cooked mesh containing optimized LOD0 geometry, coarser LOD index buffers, and meshlets.
struct CookedMesh {
    Aabb bounds{};
    std::vector<Vertex> vertices;
    std::vector<SkinVertex> skin;
    std::vector<std::uint32_t> indices;  // LOD0
    std::vector<MeshLod> lods;
    std::vector<CookedMeshlet> meshlets;
    std::vector<std::uint32_t> meshletVertices;
    std::vector<std::uint8_t> meshletTriangles;
};

/// Runs vertex/cache optimization, builds LODs, and generates bounded meshlets for LOD0.
[[nodiscard]] CookedMesh cookMesh(const MeshData& source);

/// Writes a binary `.lmesh` file atomically with header metadata and an FNV-1a payload checksum.
[[nodiscard]] Result<> writeCookedMesh(const std::filesystem::path& path, const CookedMesh& mesh);

/// Reads and validates a binary `.lmesh` file, rejecting truncated or checksum-corrupted files.
[[nodiscard]] Result<CookedMesh> readCookedMesh(const std::filesystem::path& path);

} // namespace lumora
