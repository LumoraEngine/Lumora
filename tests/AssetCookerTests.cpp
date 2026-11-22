#include "assets/AssetDatabase.hpp"
#include "assets/MeshCooker.hpp"
#include "assets/Primitives.hpp"
#include "GltfTestUtils.hpp"
#include "TestUtils.hpp"

#include <glm/geometric.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace lumora;

using TriKey = std::array<float, 9>;

std::multiset<TriKey> canonicalTriangles(const std::vector<Vertex>& vertices, const std::vector<std::uint32_t>& indices) {
    std::multiset<TriKey> out;
    for (std::size_t t = 0; t + 2 < indices.size(); t += 3) {
        std::array<glm::vec3, 3> p{
            vertices[indices[t]].position,
            vertices[indices[t + 1]].position,
            vertices[indices[t + 2]].position,
        };
        const auto first = std::ranges::min_element(p, [](const glm::vec3& a, const glm::vec3& b) {
            return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
        }) - p.begin();
        std::ranges::rotate(p, p.begin() + first);
        out.insert({p[0].x, p[0].y, p[0].z, p[1].x, p[1].y, p[1].z, p[2].x, p[2].y, p[2].z});
    }
    return out;
}

std::vector<std::uint32_t> expandMeshletTriangles(const CookedMesh& mesh) {
    std::vector<std::uint32_t> indices;
    for (const CookedMeshlet& m : mesh.meshlets) {
        for (std::uint32_t t = 0; t < m.triangleCount; ++t) {
            const std::size_t base = static_cast<std::size_t>(m.triangleOffset) + static_cast<std::size_t>(t) * 3;
            const std::uint8_t l0 = mesh.meshletTriangles[base];
            const std::uint8_t l1 = mesh.meshletTriangles[base + 1];
            const std::uint8_t l2 = mesh.meshletTriangles[base + 2];
            indices.push_back(mesh.meshletVertices[m.vertexOffset + l0]);
            indices.push_back(mesh.meshletVertices[m.vertexOffset + l1]);
            indices.push_back(mesh.meshletVertices[m.vertexOffset + l2]);
        }
    }
    return indices;
}

void writeText(const std::filesystem::path& path, std::string_view text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void expectCookedEqual(const CookedMesh& a, const CookedMesh& b) {
    EXPECT_EQ(a.bounds.min, b.bounds.min);
    EXPECT_EQ(a.bounds.max, b.bounds.max);
    ASSERT_EQ(a.vertices.size(), b.vertices.size());
    for (std::size_t i = 0; i < a.vertices.size(); ++i) {
        EXPECT_EQ(a.vertices[i].position, b.vertices[i].position);
        EXPECT_EQ(a.vertices[i].normal, b.vertices[i].normal);
        EXPECT_EQ(a.vertices[i].uv, b.vertices[i].uv);
        EXPECT_EQ(a.vertices[i].tangent, b.vertices[i].tangent);
        EXPECT_EQ(a.vertices[i].uv1, b.vertices[i].uv1);
    }
    ASSERT_EQ(a.skin.size(), b.skin.size());
    for (std::size_t i = 0; i < a.skin.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(a.skin[i].joints[j], b.skin[i].joints[j]);
        }
        EXPECT_EQ(a.skin[i].weights, b.skin[i].weights);
    }
    EXPECT_EQ(a.indices, b.indices);
    ASSERT_EQ(a.lods.size(), b.lods.size());
    for (std::size_t i = 0; i < a.lods.size(); ++i) {
        EXPECT_FLOAT_EQ(a.lods[i].error, b.lods[i].error);
        EXPECT_EQ(a.lods[i].indices, b.lods[i].indices);
    }
    EXPECT_EQ(a.meshlets, b.meshlets);
    EXPECT_EQ(a.meshletVertices, b.meshletVertices);
    EXPECT_EQ(a.meshletTriangles, b.meshletTriangles);
}

TEST(AssetCooker, CookMeshAndLmeshRoundtripPreservesAllStreams) {
    MeshData sphere = makeUvSphere(1.5f, 64, 32);
    sphere.skin.resize(sphere.vertices.size());
    for (std::size_t i = 0; i < sphere.vertices.size(); ++i) {
        SkinVertex s;
        s.joints[0] = static_cast<std::uint16_t>(sphere.vertices[i].position.y >= 0.0f ? 1 : 2);
        s.joints[1] = 3;
        s.weights = {0.75f, 0.25f, 0.0f, 0.0f};
        sphere.skin[i] = s;
    }

    const CookedMesh cooked = cookMesh(sphere);
    ASSERT_FALSE(cooked.vertices.empty());
    ASSERT_FALSE(cooked.indices.empty());
    EXPECT_EQ(cooked.skin.size(), cooked.vertices.size());
    EXPECT_GE(cooked.lods.size(), 2u);
    EXPECT_FALSE(cooked.meshlets.empty());
    EXPECT_FALSE(cooked.meshletVertices.empty());
    EXPECT_FALSE(cooked.meshletTriangles.empty());

    EXPECT_NEAR(cooked.bounds.min.x, -1.5f, 1e-3f);
    EXPECT_NEAR(cooked.bounds.max.x, 1.5f, 1e-3f);
    EXPECT_NEAR(cooked.bounds.min.y, -1.5f, 1e-3f);
    EXPECT_NEAR(cooked.bounds.max.y, 1.5f, 1e-3f);

    for (const CookedMeshlet& m : cooked.meshlets) {
        EXPECT_GT(m.vertexCount, 0u);
        EXPECT_LE(m.vertexCount, kMaxCookedMeshletVertices);
        EXPECT_GT(m.triangleCount, 0u);
        EXPECT_LE(m.triangleCount, kMaxCookedMeshletTriangles);
        EXPECT_GT(m.boundingSphere.w, 0.0f);
    }

    const std::vector<std::uint32_t> expanded = expandMeshletTriangles(cooked);
    EXPECT_EQ(expanded.size(), cooked.indices.size());
    EXPECT_EQ(canonicalTriangles(cooked.vertices, expanded), canonicalTriangles(cooked.vertices, cooked.indices));

    test::TempDir temp;
    const std::filesystem::path lmeshPath = temp.path() / "Cooked" / "sphere.lmesh";
    const auto writeResult = writeCookedMesh(lmeshPath, cooked);
    ASSERT_TRUE(writeResult.has_value()) << writeResult.error().message;

    const auto loaded = readCookedMesh(lmeshPath);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    expectCookedEqual(cooked, *loaded);
}

TEST(AssetCooker, CorruptedLmeshTriggersChecksumAndValidationFailure) {
    const CookedMesh cooked = cookMesh(makeUvSphere(1.0f, 32, 16));
    ASSERT_FALSE(cooked.meshlets.empty());

    test::TempDir temp;
    const std::filesystem::path validPath = temp.path() / "valid.lmesh";
    ASSERT_TRUE(writeCookedMesh(validPath, cooked).has_value());

    std::ifstream in(validPath, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.good());
    const auto size = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::vector<char> bytes(size);
    ASSERT_TRUE(in.read(bytes.data(), static_cast<std::streamsize>(size)).good());
    in.close();
    ASSERT_GT(bytes.size(), 128u);

    {
        std::vector<char> corrupted = bytes;
        corrupted[corrupted.size() - 1] ^= 0x5A;
        const auto badPayloadPath = temp.path() / "bad_payload_tail.lmesh";
        std::ofstream out(badPayloadPath, std::ios::binary);
        out.write(corrupted.data(), static_cast<std::streamsize>(corrupted.size()));
        out.close();
        const auto res = readCookedMesh(badPayloadPath);
        EXPECT_FALSE(res.has_value());
    }

    {
        std::vector<char> corrupted = bytes;
        corrupted[corrupted.size() / 2] ^= 0x01;
        const auto badMidPath = temp.path() / "bad_payload_mid.lmesh";
        std::ofstream out(badMidPath, std::ios::binary);
        out.write(corrupted.data(), static_cast<std::streamsize>(corrupted.size()));
        out.close();
        const auto res = readCookedMesh(badMidPath);
        EXPECT_FALSE(res.has_value());
    }

    {
        std::vector<char> badMagic = bytes;
        badMagic[0] = 'X';
        const auto badMagicPath = temp.path() / "bad_magic.lmesh";
        std::ofstream out(badMagicPath, std::ios::binary);
        out.write(badMagic.data(), static_cast<std::streamsize>(badMagic.size()));
        out.close();
        EXPECT_FALSE(readCookedMesh(badMagicPath).has_value());
    }

    {
        const auto truncatedPath = temp.path() / "truncated.lmesh";
        std::ofstream out(truncatedPath, std::ios::binary);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
        out.close();
        EXPECT_FALSE(readCookedMesh(truncatedPath).has_value());
    }

    EXPECT_FALSE(readCookedMesh(temp.path() / "missing.lmesh").has_value());
}

TEST(AssetCooker, EmptyAndSmallMeshCookRoundtrip) {
    const CookedMesh emptyCooked = cookMesh(MeshData{});
    EXPECT_TRUE(emptyCooked.vertices.empty());
    EXPECT_TRUE(emptyCooked.indices.empty());
    EXPECT_TRUE(emptyCooked.meshlets.empty());

    const CookedMesh cubeCooked = cookMesh(makeCube(2.0f));
    EXPECT_EQ(cubeCooked.vertices.size(), 24u);
    EXPECT_EQ(cubeCooked.indices.size(), 36u);
    EXPECT_TRUE(cubeCooked.lods.empty());
    ASSERT_EQ(cubeCooked.meshlets.size(), 1u);
    EXPECT_EQ(cubeCooked.meshlets[0].triangleCount, 12u);

    test::TempDir temp;
    ASSERT_TRUE(writeCookedMesh(temp.path() / "empty.lmesh", emptyCooked).has_value());
    const auto emptyLoaded = readCookedMesh(temp.path() / "empty.lmesh");
    ASSERT_TRUE(emptyLoaded.has_value());
    expectCookedEqual(emptyCooked, *emptyLoaded);

    ASSERT_TRUE(writeCookedMesh(temp.path() / "cube.lmesh", cubeCooked).has_value());
    const auto cubeLoaded = readCookedMesh(temp.path() / "cube.lmesh");
    ASSERT_TRUE(cubeLoaded.has_value());
    expectCookedEqual(cubeCooked, *cubeLoaded);
}

TEST(AssetDatabase, ScanSaveLoadIncrementalStalenessAndCooking) {
    test::TempDir temp;
    const std::filesystem::path projectRoot = temp.path() / "Project";
    const std::filesystem::path outputRoot = temp.path() / "Output";

    std::filesystem::create_directories(projectRoot / "Content" / "Models");
    std::filesystem::create_directories(projectRoot / "Content" / "Materials");
    std::filesystem::create_directories(projectRoot / "Content" / "Textures");
    std::filesystem::create_directories(projectRoot / "Scripts");
    std::filesystem::create_directories(projectRoot / "Saved");
    std::filesystem::create_directories(projectRoot / "Cooked");

    test::writeCubeGltf(projectRoot / "Content" / "Models" / "cube.gltf", false);
    test::writeSphereGltf(projectRoot / "Content" / "Models" / "sphere.gltf", 32, 16);
    writeText(projectRoot / "Content" / "Materials" / "red.lmat", R"({"baseColor":[0.9,0.1,0.1]})");
    writeText(projectRoot / "Content" / "Textures" / "dummy.png", "PNG_PLACEHOLDER_DATA");
    writeText(projectRoot / "Scripts" / "spin.lua", "function update(dt) end\n");
    writeText(projectRoot / "Saved" / "cached.txt", "ignore_me");
    writeText(projectRoot / "Cooked" / "stale.lmesh", "ignore_me");

    AssetDatabase db;
    const auto scanResult = db.scanContent(projectRoot);
    ASSERT_TRUE(scanResult.has_value()) << scanResult.error().message;
    EXPECT_EQ(*scanResult, 5u);
    EXPECT_EQ(db.size(), 5u);

    const AssetRecord* cubeRec = db.find("Content/Models/cube.gltf");
    const AssetRecord* sphereRec = db.find("Content/Models/sphere.gltf");
    const AssetRecord* matRec = db.find("Content/Materials/red.lmat");
    const AssetRecord* texRec = db.find("Content/Textures/dummy.png");
    const AssetRecord* scriptRec = db.find("Scripts/spin.lua");
    ASSERT_NE(cubeRec, nullptr);
    ASSERT_NE(sphereRec, nullptr);
    ASSERT_NE(matRec, nullptr);
    ASSERT_NE(texRec, nullptr);
    ASSERT_NE(scriptRec, nullptr);

    EXPECT_EQ(cubeRec->kind, AssetKind::Model);
    EXPECT_EQ(sphereRec->kind, AssetKind::Model);
    EXPECT_EQ(matRec->kind, AssetKind::Material);
    EXPECT_EQ(texRec->kind, AssetKind::Texture);
    EXPECT_EQ(scriptRec->kind, AssetKind::Script);
    EXPECT_GT(cubeRec->sizeBytes, 0u);
    EXPECT_NE(cubeRec->mtimeNs, 0);
    EXPECT_NE(cubeRec->contentHash, 0u);
    EXPECT_FALSE(cubeRec->cookedValid);

    const auto initialStale = db.findStaleAssets(projectRoot);
    EXPECT_EQ(initialStale.size(), 2u);

    const auto stats1 = db.cookStaleMeshes(projectRoot, outputRoot);
    ASSERT_TRUE(stats1.has_value()) << stats1.error().message;
    EXPECT_EQ(stats1->cooked, 2u);
    EXPECT_EQ(stats1->skippedUpToDate, 0u);
    EXPECT_EQ(stats1->failed, 0u);
    EXPECT_GT(stats1->bytesWritten, 0u);

    cubeRec = db.find("Content/Models/cube.gltf");
    ASSERT_NE(cubeRec, nullptr);
    EXPECT_TRUE(cubeRec->cookedValid);
    EXPECT_FALSE(cubeRec->cookedPath.empty());
    EXPECT_TRUE(std::filesystem::exists(outputRoot / cubeRec->cookedPath));

    const auto cookedCube = readCookedMesh(outputRoot / cubeRec->cookedPath);
    ASSERT_TRUE(cookedCube.has_value()) << cookedCube.error().message;
    EXPECT_EQ(cookedCube->indices.size(), 36u);

    EXPECT_TRUE(db.findStaleAssets(projectRoot).empty());

    const std::filesystem::path dbPath = projectRoot / "Saved" / "AssetDatabase.json";
    ASSERT_TRUE(db.save(dbPath).has_value());

    AssetDatabase loadedDb;
    ASSERT_TRUE(loadedDb.load(dbPath).has_value());
    EXPECT_EQ(loadedDb.records(), db.records());
    EXPECT_TRUE(loadedDb.findStaleAssets(projectRoot).empty());

    // Re-scan and re-cook without modifying files: everything unchanged is skipped.
    const auto rescanResult = loadedDb.scanContent(projectRoot);
    ASSERT_TRUE(rescanResult.has_value());
    EXPECT_EQ(*rescanResult, 5u);
    EXPECT_EQ(loadedDb.records(), db.records());

    const auto stats2 = loadedDb.cookStaleMeshes(projectRoot, outputRoot);
    ASSERT_TRUE(stats2.has_value());
    EXPECT_EQ(stats2->cooked, 0u);
    EXPECT_EQ(stats2->skippedUpToDate, 2u);
    EXPECT_EQ(stats2->failed, 0u);
    EXPECT_EQ(stats2->bytesWritten, 0u);

    // Modify one model and one same-size material on disk.
    test::writeSphereGltf(projectRoot / "Content" / "Models" / "cube.gltf", 16, 8);
    writeText(projectRoot / "Content" / "Materials" / "red.lmat", R"({"baseColor":[0.1,0.9,0.1]})");

    const auto staleAfterEdit = loadedDb.findStaleAssets(projectRoot);
    EXPECT_EQ(staleAfterEdit,
              (std::vector<std::string>{"Content/Materials/red.lmat", "Content/Models/cube.gltf"}));

    const auto stats3 = loadedDb.cookStaleMeshes(projectRoot, outputRoot);
    ASSERT_TRUE(stats3.has_value());
    EXPECT_EQ(stats3->cooked, 1u);
    EXPECT_EQ(stats3->skippedUpToDate, 1u);
    EXPECT_EQ(stats3->failed, 0u);
    EXPECT_GT(stats3->bytesWritten, 0u);

    // Missing cooked output file triggers re-cook for that model only.
    const AssetRecord* updatedCube = loadedDb.find("Content/Models/cube.gltf");
    ASSERT_NE(updatedCube, nullptr);
    std::filesystem::remove(outputRoot / updatedCube->cookedPath);

    const auto stats4 = loadedDb.cookStaleMeshes(projectRoot, outputRoot);
    ASSERT_TRUE(stats4.has_value());
    EXPECT_EQ(stats4->cooked, 1u);
    EXPECT_EQ(stats4->skippedUpToDate, 1u);
    EXPECT_EQ(stats4->failed, 0u);
    EXPECT_TRUE(std::filesystem::exists(outputRoot / updatedCube->cookedPath));
}

} // namespace
