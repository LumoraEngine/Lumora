// Levels of detail for meshes: simplified index lists over the mesh's own vertices.
//
// A level is made by edge collapses (meshoptimizer's simplifier) under an error bound. Vertices
// on the mesh's open borders never move, so the parts a model is made of still meet at their
// seams in every level; UV and normal seams are kept too. The levels reuse the original vertex
// buffer (and skin weights), so only an index buffer per level is stored.
#pragma once

#include "assets/MeshData.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace lumora {

struct MeshLod {
    std::vector<std::uint32_t> indices;  // into the original vertices
    float error = 0.0f;                  // how far the surface moved, relative to the mesh's largest side
};

/// One simplified index list with at most about `targetTriangles` triangles, moving the surface
/// by at most `maxError` (relative to the mesh's largest side). `error` receives what it moved.
[[nodiscard]] std::vector<std::uint32_t> simplifyMesh(const MeshData& mesh, std::size_t targetTriangles, float maxError,
                                                      float* error = nullptr);

/// Up to `maxLevels` coarser levels (LOD1, LOD2, ...), each with at most ~60 % of the previous
/// level's triangles; none for meshes under `minTriangles` or ones that cannot get smaller.
[[nodiscard]] std::vector<MeshLod> buildLods(const MeshData& mesh, int maxLevels = 3, std::size_t minTriangles = 256);

/// The level to draw and how far it has faded towards the next finer one (LOD blending).
struct LodChoice {
    int level = 0;      // 0 = full detail
    float fade = 0.0f;  // 0..1: share of level - 1 drawn in its place (dithered); 0 at level 0
};

/// The coarsest level whose error, projected on the screen, stays under `pixels` / `bias` (bias
/// > 1 keeps detail longer; 0 = always full detail). Over the last 30 % below that limit the level
/// fades into the finer one, so a switch is a cross-fade rather than a pop.
/// `worldErrors[i]` is level i+1's error in world units; `pixelsPerUnit` is how many pixels one
/// world unit at distance 1 covers (screen height / (2 tan(fovY / 2))).
[[nodiscard]] LodChoice selectLod(std::span<const float> worldErrors, float distance, float pixelsPerUnit, float bias = 1.0f,
                                  float pixels = 1.0f);

/// Fraction of the screen height covered by a sphere seen from `distance` with a vertical
/// field of view of `fovY` radians.
[[nodiscard]] float screenCoverage(float radius, float distance, float fovY);

} // namespace lumora
