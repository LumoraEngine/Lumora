#include "assets/MeshCooker.hpp"

#include "assets/MeshOptimize.hpp"
#include "core/FileSystem.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace lumora {

namespace {

constexpr char kLmeshMagic[4] = {'L', 'M', 'S', 'H'};
constexpr std::uint32_t kLmeshVersion = 1;
constexpr float kMeshletConeWeight = 0.5f;

static_assert(sizeof(Vertex) == 56, "Vertex must remain tightly packed for .lmesh serialization");
static_assert(sizeof(SkinVertex) == 24, "SkinVertex must remain tightly packed for .lmesh serialization");
static_assert(sizeof(CookedMeshlet) == 60, "CookedMeshlet must remain tightly packed for .lmesh serialization");

struct LmeshHeader {
    char magic[4] = {'L', 'M', 'S', 'H'};
    std::uint32_t version = kLmeshVersion;
    std::uint32_t vertexCount = 0;
    std::uint32_t skinCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t lodCount = 0;
    std::uint32_t meshletCount = 0;
    std::uint32_t meshletVertexCount = 0;
    std::uint32_t meshletTriangleCount = 0;
    std::uint32_t reserved = 0;
    float boundsMin[3] = {0.0f, 0.0f, 0.0f};
    float boundsMax[3] = {0.0f, 0.0f, 0.0f};
    std::uint64_t payloadBytes = 0;
    std::uint64_t payloadHash = 0;
};
static_assert(sizeof(LmeshHeader) == 80, "Unexpected LmeshHeader padding");

[[nodiscard]] std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const std::byte b : data) {
        hash ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(b));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

void appendRaw(std::vector<std::byte>& dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    const std::size_t at = dst.size();
    dst.resize(at + bytes);
    std::memcpy(dst.data() + at, src, bytes);
}

template <class T>
void appendValue(std::vector<std::byte>& dst, const T& value) {
    appendRaw(dst, &value, sizeof(T));
}

template <class T>
void appendSpan(std::vector<std::byte>& dst, const std::vector<T>& items) {
    appendRaw(dst, items.data(), items.size() * sizeof(T));
}

template <class T>
[[nodiscard]] bool readValue(const std::byte*& cur, const std::byte* end, T& out) noexcept {
    if (static_cast<std::size_t>(end - cur) < sizeof(T)) return false;
    std::memcpy(&out, cur, sizeof(T));
    cur += sizeof(T);
    return true;
}

template <class T>
[[nodiscard]] bool readVector(const std::byte*& cur, const std::byte* end, std::size_t count, std::vector<T>& out) {
    const std::size_t remaining = static_cast<std::size_t>(end - cur);
    if (count > remaining / sizeof(T)) return false;
    out.resize(count);
    const std::size_t bytes = count * sizeof(T);
    if (bytes > 0) {
        std::memcpy(out.data(), cur, bytes);
        cur += bytes;
    }
    return true;
}

[[nodiscard]] bool isFiniteVec3(const glm::vec3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] bool fitsUint32(std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
}

} // namespace

MeshletData toMeshletData(const CookedMesh& cooked) {
    MeshletData out;
    out.meshlets.reserve(cooked.meshlets.size());
    for (const CookedMeshlet& m : cooked.meshlets) {
        out.meshlets.push_back(toMeshlet(m));
    }
    out.meshletVertices = cooked.meshletVertices;
    out.meshletTriangles = cooked.meshletTriangles;
    return out;
}

CookedMesh cookMesh(const MeshData& source) {
    CookedMesh cooked;
    if (source.vertices.empty() || source.indices.size() < 3 || source.indices.size() % 3 != 0) {
        cooked.bounds = computeBounds(source.vertices);
        cooked.vertices = source.vertices;
        cooked.skin = source.skin;
        return cooked;
    }
    for (const std::uint32_t i : source.indices) {
        if (i >= source.vertices.size()) {
            return cooked;
        }
    }

    MeshData working = source;
    optimizeMesh(working);
    working.bounds = computeBounds(working.vertices);

    cooked.bounds = working.bounds;
    cooked.lods = buildLods(working);

    MeshletData built = buildMeshlets(working, kMaxCookedMeshletVertices, kMaxCookedMeshletTriangles,
                                      kMeshletConeWeight);
    cooked.vertices = std::move(working.vertices);
    cooked.skin = std::move(working.skin);
    cooked.indices = std::move(working.indices);

    cooked.meshlets.reserve(built.meshlets.size());
    for (const Meshlet& m : built.meshlets) {
        cooked.meshlets.push_back(toCookedMeshlet(m));
    }
    cooked.meshletVertices = std::move(built.meshletVertices);
    cooked.meshletTriangles = std::move(built.meshletTriangles);
    return cooked;
}

Result<> writeCookedMesh(const std::filesystem::path& path, const CookedMesh& mesh) {
    if (!fitsUint32(mesh.vertices.size()) || !fitsUint32(mesh.skin.size()) || !fitsUint32(mesh.indices.size()) ||
        !fitsUint32(mesh.lods.size()) || !fitsUint32(mesh.meshlets.size()) ||
        !fitsUint32(mesh.meshletVertices.size()) || !fitsUint32(mesh.meshletTriangles.size())) {
        return makeError("mesh '{}' exceeds 32-bit .lmesh count limits", path.string());
    }
    for (const MeshLod& lod : mesh.lods) {
        if (!fitsUint32(lod.indices.size())) {
            return makeError("mesh '{}' LOD exceeds 32-bit .lmesh count limits", path.string());
        }
    }

    std::size_t expectedPayload = mesh.vertices.size() * sizeof(Vertex) +
                                  mesh.skin.size() * sizeof(SkinVertex) +
                                  mesh.indices.size() * sizeof(std::uint32_t) +
                                  mesh.meshlets.size() * sizeof(CookedMeshlet) +
                                  mesh.meshletVertices.size() * sizeof(std::uint32_t) +
                                  mesh.meshletTriangles.size() * sizeof(std::uint8_t);
    for (const MeshLod& lod : mesh.lods) {
        expectedPayload += sizeof(float) + sizeof(std::uint32_t) + lod.indices.size() * sizeof(std::uint32_t);
    }

    std::vector<std::byte> fileBytes;
    fileBytes.reserve(sizeof(LmeshHeader) + expectedPayload);
    fileBytes.resize(sizeof(LmeshHeader));

    appendSpan(fileBytes, mesh.vertices);
    appendSpan(fileBytes, mesh.skin);
    appendSpan(fileBytes, mesh.indices);
    for (const MeshLod& lod : mesh.lods) {
        appendValue(fileBytes, lod.error);
        const auto lodIndexCount = static_cast<std::uint32_t>(lod.indices.size());
        appendValue(fileBytes, lodIndexCount);
        appendSpan(fileBytes, lod.indices);
    }
    appendSpan(fileBytes, mesh.meshlets);
    appendSpan(fileBytes, mesh.meshletVertices);
    appendSpan(fileBytes, mesh.meshletTriangles);

    const std::size_t payloadSize = fileBytes.size() - sizeof(LmeshHeader);
    const std::span<const std::byte> payloadSpan(fileBytes.data() + sizeof(LmeshHeader), payloadSize);

    LmeshHeader header;
    std::memcpy(header.magic, kLmeshMagic, sizeof(kLmeshMagic));
    header.version = kLmeshVersion;
    header.vertexCount = static_cast<std::uint32_t>(mesh.vertices.size());
    header.skinCount = static_cast<std::uint32_t>(mesh.skin.size());
    header.indexCount = static_cast<std::uint32_t>(mesh.indices.size());
    header.lodCount = static_cast<std::uint32_t>(mesh.lods.size());
    header.meshletCount = static_cast<std::uint32_t>(mesh.meshlets.size());
    header.meshletVertexCount = static_cast<std::uint32_t>(mesh.meshletVertices.size());
    header.meshletTriangleCount = static_cast<std::uint32_t>(mesh.meshletTriangles.size());
    header.reserved = 0;
    header.boundsMin[0] = mesh.bounds.min.x;
    header.boundsMin[1] = mesh.bounds.min.y;
    header.boundsMin[2] = mesh.bounds.min.z;
    header.boundsMax[0] = mesh.bounds.max.x;
    header.boundsMax[1] = mesh.bounds.max.y;
    header.boundsMax[2] = mesh.bounds.max.z;
    header.payloadBytes = static_cast<std::uint64_t>(payloadSize);
    header.payloadHash = fnv1a64(payloadSpan);

    std::memcpy(fileBytes.data(), &header, sizeof(LmeshHeader));
    return fs::writeFileAtomic(path, fileBytes);
}

Result<CookedMesh> readCookedMesh(const std::filesystem::path& path) {
    auto fileBytes = fs::readBinaryFile(path);
    if (!fileBytes) {
        return std::unexpected(fileBytes.error());
    }
    if (fileBytes->size() < sizeof(LmeshHeader)) {
        return makeError("{}: truncated .lmesh header", path.string());
    }

    LmeshHeader header;
    std::memcpy(&header, fileBytes->data(), sizeof(LmeshHeader));

    if (std::memcmp(header.magic, kLmeshMagic, sizeof(kLmeshMagic)) != 0) {
        return makeError("{}: invalid .lmesh magic", path.string());
    }
    if (header.version != kLmeshVersion) {
        return makeError("{}: unsupported .lmesh version {}", path.string(), header.version);
    }
    if (header.reserved != 0) {
        return makeError("{}: corrupted .lmesh header", path.string());
    }

    const std::size_t actualPayloadBytes = fileBytes->size() - sizeof(LmeshHeader);
    if (header.payloadBytes != static_cast<std::uint64_t>(actualPayloadBytes)) {
        return makeError("{}: .lmesh payload size mismatch (expected {}, got {})",
                         path.string(), header.payloadBytes, actualPayloadBytes);
    }

    const std::span<const std::byte> payload(fileBytes->data() + sizeof(LmeshHeader), actualPayloadBytes);
    const std::uint64_t computedHash = fnv1a64(payload);
    if (computedHash != header.payloadHash) {
        return makeError("{}: .lmesh checksum mismatch (expected {:#018x}, got {:#018x})",
                         path.string(), header.payloadHash, computedHash);
    }

    CookedMesh out;
    out.bounds.min = glm::vec3(header.boundsMin[0], header.boundsMin[1], header.boundsMin[2]);
    out.bounds.max = glm::vec3(header.boundsMax[0], header.boundsMax[1], header.boundsMax[2]);
    if (!isFiniteVec3(out.bounds.min) || !isFiniteVec3(out.bounds.max)) {
        return makeError("{}: invalid non-finite bounds in .lmesh header", path.string());
    }
    if (header.vertexCount > 0 &&
        (out.bounds.min.x > out.bounds.max.x || out.bounds.min.y > out.bounds.max.y ||
         out.bounds.min.z > out.bounds.max.z)) {
        return makeError("{}: inverted bounds in .lmesh header", path.string());
    }

    const std::byte* cur = payload.data();
    const std::byte* const end = cur + payload.size();

    if (!readVector(cur, end, header.vertexCount, out.vertices)) {
        return makeError("{}: truncated vertex stream", path.string());
    }
    if (!readVector(cur, end, header.skinCount, out.skin)) {
        return makeError("{}: truncated skin stream", path.string());
    }
    if (!readVector(cur, end, header.indexCount, out.indices)) {
        return makeError("{}: truncated index stream", path.string());
    }

    const std::size_t minLodHeaderBytes = sizeof(float) + sizeof(std::uint32_t);
    if (static_cast<std::size_t>(header.lodCount) > static_cast<std::size_t>(end - cur) / minLodHeaderBytes) {
        return makeError("{}: truncated LOD table", path.string());
    }
    out.lods.reserve(header.lodCount);
    for (std::uint32_t l = 0; l < header.lodCount; ++l) {
        MeshLod lod;
        std::uint32_t lodIndexCount = 0;
        if (!readValue(cur, end, lod.error) || !readValue(cur, end, lodIndexCount)) {
            return makeError("{}: truncated LOD {} header", path.string(), l);
        }
        if (!std::isfinite(lod.error)) {
            return makeError("{}: non-finite LOD {} error", path.string(), l);
        }
        if (!readVector(cur, end, lodIndexCount, lod.indices)) {
            return makeError("{}: truncated LOD {} indices", path.string(), l);
        }
        out.lods.push_back(std::move(lod));
    }

    if (!readVector(cur, end, header.meshletCount, out.meshlets)) {
        return makeError("{}: truncated meshlet descriptors", path.string());
    }
    if (!readVector(cur, end, header.meshletVertexCount, out.meshletVertices)) {
        return makeError("{}: truncated meshlet vertex indirections", path.string());
    }
    if (!readVector(cur, end, header.meshletTriangleCount, out.meshletTriangles)) {
        return makeError("{}: truncated meshlet triangles", path.string());
    }

    if (cur != end) {
        return makeError("{}: unexpected trailing bytes in .lmesh payload", path.string());
    }

    const std::size_t vertexCount = out.vertices.size();
    for (const std::uint32_t idx : out.indices) {
        if (idx >= vertexCount) {
            return makeError("{}: LOD0 index out of vertex bounds", path.string());
        }
    }
    for (const MeshLod& lod : out.lods) {
        for (const std::uint32_t idx : lod.indices) {
            if (idx >= vertexCount) {
                return makeError("{}: LOD index out of vertex bounds", path.string());
            }
        }
    }
    for (const std::uint32_t vIdx : out.meshletVertices) {
        if (vIdx >= vertexCount) {
            return makeError("{}: meshlet vertex index out of vertex bounds", path.string());
        }
    }
    for (const CookedMeshlet& m : out.meshlets) {
        const std::uint64_t vEnd = static_cast<std::uint64_t>(m.vertexOffset) + m.vertexCount;
        const std::uint64_t tEnd =
            static_cast<std::uint64_t>(m.triangleOffset) + static_cast<std::uint64_t>(m.triangleCount) * 3ULL;
        if (vEnd > out.meshletVertices.size() || tEnd > out.meshletTriangles.size()) {
            return makeError("{}: meshlet range out of buffer bounds", path.string());
        }
        for (std::uint32_t t = 0; t < m.triangleCount * 3; ++t) {
            if (out.meshletTriangles[m.triangleOffset + t] >= m.vertexCount) {
                return makeError("{}: meshlet local triangle index out of meshlet vertex bounds", path.string());
            }
        }
    }

    return out;
}

} // namespace lumora
