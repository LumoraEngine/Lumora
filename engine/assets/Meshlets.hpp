#pragma once

#include "assets/MeshData.hpp"

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lumora {

/// Hardware mesh-shader friendly cluster limits enforced by buildMeshlets.
inline constexpr std::size_t kMaxMeshletVertices = 64;
inline constexpr std::size_t kMaxMeshletTriangles = 126;

/// GPU-friendly packed cluster descriptor (64 bytes, std430 / HLSL StructuredBuffer compatible).
struct Meshlet {
    std::uint32_t vertexOffset = 0;    // start index in MeshletData::meshletVertices
    std::uint32_t triangleOffset = 0;  // start index in MeshletData::meshletTriangles (3 uint8 per tri)
    std::uint32_t vertexCount = 0;     // <= kMaxMeshletVertices (64)
    std::uint32_t triangleCount = 0;   // <= kMaxMeshletTriangles (126)
    glm::vec4 boundingSphere{0.0f};    // xyz = local center, w = radius
    glm::vec3 coneApex{0.0f};          // local normal-cone apex for backface culling
    float coneCutoff = 1.0f;           // cos(angle / 2) threshold (>= 1 disables cone culling)
    glm::vec3 coneAxis{0.0f};          // unit normal-cone axis in mesh local space
    std::uint32_t flags = 0;           // cluster flags / 16-byte struct alignment padding
};
static_assert(sizeof(Meshlet) == 64, "GPU-friendly 64-byte Meshlet layout");
static_assert(offsetof(Meshlet, boundingSphere) == 16);
static_assert(offsetof(Meshlet, coneApex) == 32);
static_assert(offsetof(Meshlet, coneCutoff) == 44);
static_assert(offsetof(Meshlet, coneAxis) == 48);
static_assert(offsetof(Meshlet, flags) == 60);

/// Mesh partitioned into hardware-sized meshlets and their indirection buffers.
struct MeshletData {
    std::vector<Meshlet> meshlets;
    std::vector<std::uint32_t> meshletVertices;  // local vertex -> global mesh vertex index
    std::vector<std::uint8_t> meshletTriangles;  // packed 3 local uint8 indices per triangle
};

/// Partitions `mesh` into clusters bounded by `maxVertices <= kMaxMeshletVertices` and
/// `maxTriangles <= kMaxMeshletTriangles`, optimizes per-meshlet locality, and computes
/// bounding spheres and normal cones. Returns empty MeshletData if input is invalid.
[[nodiscard]] MeshletData buildMeshlets(const MeshData& mesh, std::size_t maxVertices = 64,
                                        std::size_t maxTriangles = 124, float coneWeight = 0.5f);

/// Returns true if the meshlet is completely back-facing from `cameraLocalPos`
/// using meshoptimizer's exact perspective cone-apex test:
/// `dot(normalize(coneApex - cameraLocalPos), coneAxis) >= coneCutoff`.
[[nodiscard]] bool cullMeshletCone(const Meshlet& m, const glm::vec3& cameraLocalPos) noexcept;

/// Returns true if the meshlet's bounding sphere lies completely outside any of the
/// 6 local-space inward-pointing frustum planes `(nx, ny, nz, d)` where inside is `dot(n, x) + d >= 0`.
[[nodiscard]] bool cullMeshletFrustum(const Meshlet& m,
                                      std::span<const glm::vec4, 6> localFrustumPlanes) noexcept;

/// Returns the indices of meshlets surviving both frustum and backface cone culling.
[[nodiscard]] std::vector<std::uint32_t> cullMeshlets(const MeshletData& data,
                                                      std::span<const glm::vec4, 6> localFrustumPlanes,
                                                      const glm::vec3& cameraLocalPos);

} // namespace lumora
