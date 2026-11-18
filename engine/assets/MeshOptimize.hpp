// Imported meshes made cheaper to draw without changing how they look.
#pragma once

#include "assets/MeshData.hpp"

namespace lumora {

/// Merges vertices that are identical in every attribute (position, normal, UVs, tangent, skin
/// weights, blend shape offsets), orders the triangles for the GPU's vertex cache and the
/// vertices in the order the triangles first use them. Same triangles, same look.
void optimizeMesh(MeshData& mesh);

} // namespace lumora
