#include "assets/Meshlets.hpp"

#include <glm/geometric.hpp>
#include <meshoptimizer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lumora {

static_assert(offsetof(Vertex, position) == 0, "meshoptimizer expects float3 position at offset 0");
static_assert(sizeof(std::uint32_t) == sizeof(unsigned int), "meshoptimizer uses 32-bit unsigned int indices");

MeshletData buildMeshlets(const MeshData& mesh, std::size_t maxVertices, std::size_t maxTriangles,
                          float coneWeight) {
    if (mesh.vertices.empty() || mesh.indices.size() < 3 || (mesh.indices.size() % 3) != 0) {
        return {};
    }
    if (maxVertices == 0 || maxTriangles == 0) {
        return {};
    }

    const std::size_t vertexCount = mesh.vertices.size();
    for (const std::uint32_t index : mesh.indices) {
        if (static_cast<std::size_t>(index) >= vertexCount) {
            return {};
        }
    }

    const std::size_t clampedMaxVertices = std::clamp<std::size_t>(maxVertices, 3, kMaxMeshletVertices);
    const std::size_t cappedTriangles = std::min<std::size_t>(maxTriangles, kMaxMeshletTriangles);
    // meshopt_buildMeshlets requires max_triangles <= 512 and divisible by 4.
    const std::size_t clampedMaxTriangles = std::max<std::size_t>(4, (cappedTriangles / 4) * 4);
    const float clampedConeWeight = std::isfinite(coneWeight) ? std::clamp(coneWeight, 0.0f, 1.0f) : 0.0f;

    const std::size_t maxMeshlets =
        meshopt_buildMeshletsBound(mesh.indices.size(), clampedMaxVertices, clampedMaxTriangles);
    if (maxMeshlets == 0) {
        return {};
    }

    std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
    MeshletData result;
    result.meshletVertices.resize(maxMeshlets * clampedMaxVertices);
    result.meshletTriangles.resize(maxMeshlets * clampedMaxTriangles * 3, 0);

    const std::size_t meshletCount = meshopt_buildMeshlets(
        rawMeshlets.data(),
        reinterpret_cast<unsigned int*>(result.meshletVertices.data()),
        result.meshletTriangles.data(),
        reinterpret_cast<const unsigned int*>(mesh.indices.data()),
        mesh.indices.size(),
        &mesh.vertices[0].position.x,
        vertexCount,
        sizeof(Vertex),
        clampedMaxVertices,
        clampedMaxTriangles,
        clampedConeWeight);

    if (meshletCount == 0) {
        return {};
    }

    rawMeshlets.resize(meshletCount);
    const meshopt_Meshlet& last = rawMeshlets.back();
    result.meshletVertices.resize(static_cast<std::size_t>(last.vertex_offset) + last.vertex_count);
    const std::size_t alignedLastTriBytes =
        (static_cast<std::size_t>(last.triangle_count) * 3u + 3u) & ~std::size_t{3};
    result.meshletTriangles.resize(static_cast<std::size_t>(last.triangle_offset) + alignedLastTriBytes);

    result.meshlets.reserve(meshletCount);
    for (const meshopt_Meshlet& raw : rawMeshlets) {
        auto* const verts = reinterpret_cast<unsigned int*>(&result.meshletVertices[raw.vertex_offset]);
        auto* const tris = &result.meshletTriangles[raw.triangle_offset];

        meshopt_optimizeMeshlet(verts, tris, raw.triangle_count, raw.vertex_count);

        const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
            verts,
            tris,
            raw.triangle_count,
            &mesh.vertices[0].position.x,
            vertexCount,
            sizeof(Vertex));

        Meshlet m{};
        m.vertexOffset = raw.vertex_offset;
        m.triangleOffset = raw.triangle_offset;
        m.vertexCount = raw.vertex_count;
        m.triangleCount = raw.triangle_count;
        m.boundingSphere = glm::vec4(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius);
        m.coneApex = glm::vec3(bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2]);
        m.coneCutoff = bounds.cone_cutoff;
        m.coneAxis = glm::vec3(bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2]);
        m.flags = 0u;
        result.meshlets.push_back(m);
    }

    return result;
}

bool cullMeshletCone(const Meshlet& m, const glm::vec3& cameraLocalPos) noexcept {
    if (m.coneCutoff >= 1.0f) {
        return false;
    }
    const float axisLen = glm::length(m.coneAxis);
    if (!(axisLen > 1e-12f)) {
        return false;
    }
    const glm::vec3 toApex = m.coneApex - cameraLocalPos;
    const float dist = glm::length(toApex);
    if (!(dist > 1e-12f)) {
        return false;
    }
    const glm::vec3 viewDir = toApex / dist;
    const glm::vec3 axis = m.coneAxis / axisLen;
    return glm::dot(viewDir, axis) >= m.coneCutoff;
}

bool cullMeshletFrustum(const Meshlet& m, std::span<const glm::vec4, 6> localFrustumPlanes) noexcept {
    const glm::vec3 center(m.boundingSphere);
    const float radius = std::max(m.boundingSphere.w, 0.0f);
    for (const glm::vec4& plane : localFrustumPlanes) {
        const glm::vec3 normal(plane);
        const float len = glm::length(normal);
        if (!(len > 1e-12f)) {
            continue;
        }
        if (glm::dot(normal, center) + plane.w < -radius * len) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint32_t> cullMeshlets(const MeshletData& data,
                                        std::span<const glm::vec4, 6> localFrustumPlanes,
                                        const glm::vec3& cameraLocalPos) {
    std::vector<std::uint32_t> visible;
    visible.reserve(data.meshlets.size());
    for (std::size_t i = 0; i < data.meshlets.size(); ++i) {
        const Meshlet& m = data.meshlets[i];
        if (!cullMeshletFrustum(m, localFrustumPlanes) && !cullMeshletCone(m, cameraLocalPos)) {
            visible.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return visible;
}

} // namespace lumora
