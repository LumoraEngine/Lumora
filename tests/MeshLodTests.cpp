#include "assets/MeshLod.hpp"
#include "assets/MeshOptimize.hpp"
#include "assets/Primitives.hpp"

#include <glm/geometric.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>

using namespace lumora;

TEST(MeshLod, LevelsShrinkAndStayOnTheSurface) {
    const MeshData sphere = makeUvSphere(1.0f, 128, 64);
    const std::size_t full = sphere.indices.size() / 3;
    const auto lods = buildLods(sphere);
    ASSERT_GE(lods.size(), 2u);
    std::size_t previous = full;
    float previousError = 0.0f;
    for (const MeshLod& lod : lods) {
        const std::size_t triangles = lod.indices.size() / 3;
        EXPECT_LE(static_cast<double>(triangles), previous * 0.6 + 1) << "each level clearly smaller";
        EXPECT_GE(triangles, 8u);
        EXPECT_GE(lod.error, previousError) << "coarser levels move the surface more";
        EXPECT_LT(lod.error, 0.2f);
        previous = triangles;
        previousError = lod.error;
        for (std::uint32_t i : lod.indices) {
            ASSERT_LT(i, sphere.vertices.size()) << "indices point into the original vertices";
            EXPECT_NEAR(glm::length(sphere.vertices[i].position), 1.0f, 1e-4f) << "levels use the mesh's own vertices";
        }
        for (std::size_t t = 0; t < lod.indices.size(); t += 3) {
            EXPECT_NE(lod.indices[t], lod.indices[t + 1]);
            EXPECT_NE(lod.indices[t + 1], lod.indices[t + 2]);
            EXPECT_NE(lod.indices[t], lod.indices[t + 2]);
        }
    }
    // Front faces still face outwards: the winding survives simplification.
    const MeshLod& coarse = lods.back();
    int outward = 0;
    const int total = static_cast<int>(coarse.indices.size() / 3);
    for (std::size_t t = 0; t < coarse.indices.size(); t += 3) {
        const glm::vec3 a = sphere.vertices[coarse.indices[t]].position;
        const glm::vec3 b = sphere.vertices[coarse.indices[t + 1]].position;
        const glm::vec3 c = sphere.vertices[coarse.indices[t + 2]].position;
        if (glm::dot(glm::cross(b - a, c - a), a + b + c) > 0.0f) ++outward;
    }
    EXPECT_GT(outward, total * 9 / 10);
}

TEST(MeshLod, OpenBordersStaySoThePartsOfAModelStillMeet) {
    // A flat grid is one part of a larger model: its edge meets the next part. Simplified, the
    // inside loses triangles but every vertex on the edge is still there, so no crack opens.
    const MeshData grid = makePlane(2.0f, 48);
    const auto lods = buildLods(grid);
    ASSERT_FALSE(lods.empty());
    std::set<std::pair<int, int>> edge;      // positions on the border, in grid steps
    std::set<std::pair<int, int>> kept;
    auto key = [](const glm::vec3& p) { return std::pair(static_cast<int>(std::lround(p.x * 24)), static_cast<int>(std::lround(p.z * 24))); };
    for (const Vertex& v : grid.vertices) {
        if (std::abs(std::abs(v.position.x) - 1.0f) < 1e-4f || std::abs(std::abs(v.position.z) - 1.0f) < 1e-4f) edge.insert(key(v.position));
    }
    ASSERT_EQ(edge.size(), 48u * 4u);
    for (const MeshLod& lod : lods) {
        EXPECT_LE(lod.indices.size(), grid.indices.size() / 2);
        kept.clear();
        for (std::uint32_t i : lod.indices) kept.insert(key(grid.vertices[i].position));
        for (const auto& e : edge) EXPECT_TRUE(kept.contains(e)) << e.first << "," << e.second;
    }
}

TEST(MeshLod, SmallMeshesGetNoLevels) {
    EXPECT_TRUE(buildLods(makePrimitive(MeshPrimitive::Cube)).empty());
}

TEST(MeshLod, SelectionKeepsTheErrorUnderAPixelAndFadesBeforeSwitching) {
    const std::array<float, 3> errors{0.01f, 0.05f, 0.2f};  // metres
    const float ppu = 1000.0f;                               // pixels per metre at 1 m
    // 5 m: every level would move the surface by 2 px or more.
    EXPECT_EQ(selectLod(errors, 5.0f, ppu).level, 0);
    // 20 m: LOD1 moves it 0.5 px, LOD2 2.5 px: LOD1, not fading yet.
    EXPECT_EQ(selectLod(errors, 20.0f, ppu).level, 1);
    EXPECT_FLOAT_EQ(selectLod(errors, 20.0f, ppu).fade, 0.0f);
    // 12 m: LOD1 at 0.83 px, past 70 % of the limit: fading into full detail.
    const LodChoice fading = selectLod(errors, 12.0f, ppu);
    EXPECT_EQ(fading.level, 1);
    EXPECT_NEAR(fading.fade, (0.01f * ppu / 12.0f - 0.7f) / 0.3f, 1e-4f);
    // Far away: the coarsest.
    EXPECT_EQ(selectLod(errors, 1000.0f, ppu).level, 3);
    EXPECT_EQ(selectLod(errors, 1000.0f, ppu, 0.0f).level, 0) << "bias 0 disables";
    EXPECT_EQ(selectLod(errors, 20.0f, ppu, 2.0f).level, 0) << "a larger bias keeps detail longer";
    EXPECT_FLOAT_EQ(screenCoverage(1.0f, 0.5f, 1.0f), 1.0f) << "inside the sphere";
    EXPECT_GT(screenCoverage(1.0f, 10.0f, 1.0f), screenCoverage(1.0f, 20.0f, 1.0f));
}

TEST(MeshOptimize, WeldsDuplicatesAndKeepsTheTrianglesAndTheirStreams) {
    // Every triangle with vertices of its own, as some exporters write them; skinned, one shape.
    const MeshData sphere = makeUvSphere(1.0f, 24, 12);
    MeshData split;
    for (std::uint32_t i : sphere.indices) {
        split.indices.push_back(static_cast<std::uint32_t>(split.vertices.size()));
        split.vertices.push_back(sphere.vertices[i]);
        SkinVertex s;
        s.joints[0] = static_cast<std::uint16_t>(sphere.vertices[i].position.y > 0.0f ? 1 : 0);
        split.skin.push_back(s);
    }
    split.morphs.resize(1);
    for (const Vertex& v : split.vertices) split.morphs[0].positions.push_back(v.position * 0.1f);
    auto triangles = [](const MeshData& m) {
        std::multiset<std::array<float, 9>> out;
        for (std::size_t t = 0; t < m.indices.size(); t += 3) {
            std::array<glm::vec3, 3> p{};
            for (int k = 0; k < 3; ++k) p[static_cast<std::size_t>(k)] = m.vertices[m.indices[t + static_cast<std::size_t>(k)]].position;
            const auto first = std::ranges::min_element(p, [](auto a, auto b) { return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z); }) - p.begin();
            std::ranges::rotate(p, p.begin() + first);  // same winding, canonical start
            out.insert({p[0].x, p[0].y, p[0].z, p[1].x, p[1].y, p[1].z, p[2].x, p[2].y, p[2].z});
        }
        return out;
    };
    const auto before = triangles(split);
    MeshData optimized = split;
    optimizeMesh(optimized);
    MeshData shared = sphere;  // the sphere's own vertices, welded the same way (its poles repeat)
    optimizeMesh(shared);
    EXPECT_LT(optimized.vertices.size(), split.vertices.size() / 4);
    EXPECT_EQ(optimized.vertices.size(), shared.vertices.size()) << "back to one vertex per corner";
    EXPECT_EQ(optimized.indices.size(), split.indices.size());
    EXPECT_EQ(triangles(optimized), before);
    ASSERT_EQ(optimized.skin.size(), optimized.vertices.size());
    ASSERT_EQ(optimized.morphs[0].positions.size(), optimized.vertices.size());
    for (std::size_t v = 0; v < optimized.vertices.size(); ++v) {
        EXPECT_EQ(optimized.skin[v].joints[0], optimized.vertices[v].position.y > 0.0f ? 1 : 0) << "skin follows its vertex";
        EXPECT_EQ(optimized.morphs[0].positions[v], optimized.vertices[v].position * 0.1f) << "shape follows its vertex";
    }
    // Vertices come in the order the triangles first use them.
    std::uint32_t next = 0;
    for (std::uint32_t i : optimized.indices) {
        EXPECT_LE(i, next);
        if (i == next) ++next;
    }
}
