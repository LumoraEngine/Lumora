#include "assets/MeshLod.hpp"

#include <meshoptimizer.h>

#include <algorithm>
#include <cmath>

namespace lumora {

std::vector<std::uint32_t> simplifyMesh(const MeshData& mesh, std::size_t targetTriangles, float maxError, float* error) {
    if (error != nullptr) *error = 0.0f;
    if (mesh.indices.size() < 3 || mesh.vertices.empty()) return mesh.indices;
    for (std::uint32_t i : mesh.indices) {
        if (i >= mesh.vertices.size()) return mesh.indices;  // malformed: keep the original
    }
    std::vector<std::uint32_t> out(mesh.indices.size());
    float moved = 0.0f;
    const std::size_t count = meshopt_simplify(out.data(), mesh.indices.data(), mesh.indices.size(), &mesh.vertices[0].position.x,
                                               mesh.vertices.size(), sizeof(Vertex), targetTriangles * 3, maxError,
                                               meshopt_SimplifyLockBorder, &moved);
    out.resize(count);
    meshopt_optimizeVertexCache(out.data(), out.data(), out.size(), mesh.vertices.size());
    if (error != nullptr) *error = moved;
    return out;
}

std::vector<MeshLod> buildLods(const MeshData& mesh, int maxLevels, std::size_t minTriangles) {
    std::vector<MeshLod> levels;
    const std::size_t full = mesh.indices.size() / 3;
    if (full < minTriangles) return levels;
    std::size_t previous = full;
    // Each level aims at half the triangles of the one before, allowing the surface to move more.
    float maxError = 0.01f;
    for (int level = 1; level <= maxLevels; ++level, maxError *= 2.5f) {
        float error = 0.0f;
        std::vector<std::uint32_t> indices = simplifyMesh(mesh, previous / 2, maxError, &error);
        const std::size_t triangles = indices.size() / 3;
        if (triangles < 8 || static_cast<double>(triangles) > static_cast<double>(previous) * 0.6) break;
        levels.push_back({std::move(indices), error});
        previous = triangles;
    }
    return levels;
}

LodChoice selectLod(std::span<const float> worldErrors, float distance, float pixelsPerUnit, float bias, float pixels) {
    LodChoice choice;
    if (worldErrors.empty() || !(bias > 0.0f) || !(pixelsPerUnit > 0.0f)) return choice;
    const float limit = pixels / bias;
    auto onScreen = [&](float worldError) { return worldError * pixelsPerUnit / std::max(distance, 1e-4f); };
    for (std::size_t i = worldErrors.size(); i-- > 0;) {
        const float shown = onScreen(worldErrors[i]);
        if (shown >= limit) continue;
        choice.level = static_cast<int>(i) + 1;
        choice.fade = std::clamp((shown / limit - 0.7f) / 0.3f, 0.0f, 1.0f);
        return choice;
    }
    return choice;
}

float screenCoverage(float radius, float distance, float fovY) {
    const float tanHalf = std::tan(std::max(fovY, 1e-3f) * 0.5f);
    if (distance <= radius) return 1.0f;
    return std::clamp(radius / (distance * tanHalf), 0.0f, 1.0f);
}

} // namespace lumora
