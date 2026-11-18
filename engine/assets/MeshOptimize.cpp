#include "assets/MeshOptimize.hpp"

#include <meshoptimizer.h>

#include <vector>

namespace lumora {

namespace {

template <class T>
void remap(std::vector<T>& values, const std::vector<unsigned int>& table, std::size_t count) {
    if (values.empty()) return;
    std::vector<T> out(count);
    meshopt_remapVertexBuffer(out.data(), values.data(), values.size(), sizeof(T), table.data());
    values = std::move(out);
}

void remapAll(MeshData& mesh, const std::vector<unsigned int>& table, std::size_t count) {
    const std::size_t before = mesh.vertices.size();
    remap(mesh.vertices, table, count);
    if (mesh.skin.size() == before) remap(mesh.skin, table, count);
    for (MorphTarget& m : mesh.morphs) {
        if (m.positions.size() == before) remap(m.positions, table, count);
        if (m.normals.size() == before) remap(m.normals, table, count);
        if (m.tangents.size() == before) remap(m.tangents, table, count);
    }
}

} // namespace

void optimizeMesh(MeshData& mesh) {
    const std::size_t count = mesh.vertices.size();
    if (count == 0 || mesh.indices.size() < 3 || mesh.indices.size() % 3 != 0) return;
    for (std::uint32_t i : mesh.indices) {
        if (i >= count) return;  // malformed: leave it as it is
    }
    // Welding: a vertex is a duplicate only when every stream the mesh has matches.
    std::vector<meshopt_Stream> streams{{mesh.vertices.data(), sizeof(Vertex), sizeof(Vertex)}};
    if (mesh.skin.size() == count) streams.push_back({mesh.skin.data(), sizeof(SkinVertex), sizeof(SkinVertex)});
    for (const MorphTarget& m : mesh.morphs) {
        if (m.positions.size() == count) streams.push_back({m.positions.data(), sizeof(glm::vec3), sizeof(glm::vec3)});
        if (m.normals.size() == count) streams.push_back({m.normals.data(), sizeof(glm::vec3), sizeof(glm::vec3)});
        if (m.tangents.size() == count) streams.push_back({m.tangents.data(), sizeof(glm::vec3), sizeof(glm::vec3)});
    }
    std::vector<unsigned int> table(count);
    const std::size_t unique = meshopt_generateVertexRemapMulti(table.data(), mesh.indices.data(), mesh.indices.size(), count,
                                                                streams.data(), streams.size());
    std::vector<std::uint32_t> indices(mesh.indices.size());
    meshopt_remapIndexBuffer(indices.data(), mesh.indices.data(), mesh.indices.size(), table.data());
    remapAll(mesh, table, unique);  // unused vertices are dropped too

    meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), unique);
    meshopt_optimizeVertexFetchRemap(table.data(), indices.data(), indices.size(), unique);
    meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), table.data());
    remapAll(mesh, table, unique);
    mesh.indices = std::move(indices);
}

} // namespace lumora
