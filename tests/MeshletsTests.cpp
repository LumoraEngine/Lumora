#include "assets/Meshlets.hpp"
#include "assets/Primitives.hpp"

#include <glm/geometric.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <tuple>
#include <vector>

using namespace lumora;

namespace {

using CanonicalTriangle = std::array<std::uint32_t, 3>;

CanonicalTriangle canonicalizeTriangle(std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) {
    CanonicalTriangle tri{i0, i1, i2};
    const auto minIt = std::ranges::min_element(tri);
    std::ranges::rotate(tri, minIt);  // preserves winding while normalizing start vertex
    return tri;
}

std::array<glm::vec4, 6> makeBoxFrustum(const glm::vec3& minB, const glm::vec3& maxB) {
    return {{
        {1.0f, 0.0f, 0.0f, -minB.x},   // x >= minB.x
        {-1.0f, 0.0f, 0.0f, maxB.x},   // x <= maxB.x
        {0.0f, 1.0f, 0.0f, -minB.y},   // y >= minB.y
        {0.0f, -1.0f, 0.0f, maxB.y},   // y <= maxB.y
        {0.0f, 0.0f, 1.0f, -minB.z},   // z >= minB.z
        {0.0f, 0.0f, -1.0f, maxB.z},   // z <= maxB.z
    }};
}

void verifyMeshletInvariants(const MeshData& mesh, const MeshletData& data,
                             std::size_t expectedMaxVerts, std::size_t expectedMaxTris) {
    ASSERT_FALSE(data.meshlets.empty());

    std::multiset<CanonicalTriangle> originalTriangles;
    for (std::size_t t = 0; t < mesh.indices.size(); t += 3) {
        originalTriangles.insert(
            canonicalizeTriangle(mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]));
    }

    std::multiset<CanonicalTriangle> reconstructedTriangles;
    std::size_t totalTriangles = 0;

    for (const Meshlet& m : data.meshlets) {
        EXPECT_GT(m.vertexCount, 0u);
        EXPECT_GT(m.triangleCount, 0u);
        EXPECT_LE(m.vertexCount, kMaxMeshletVertices);
        EXPECT_LE(m.triangleCount, kMaxMeshletTriangles);
        EXPECT_LE(m.vertexCount, expectedMaxVerts);
        EXPECT_LE(m.triangleCount, expectedMaxTris);

        ASSERT_LE(static_cast<std::size_t>(m.vertexOffset) + m.vertexCount,
                  data.meshletVertices.size());
        ASSERT_LE(static_cast<std::size_t>(m.triangleOffset) + static_cast<std::size_t>(m.triangleCount) * 3u,
                  data.meshletTriangles.size());

        // Every vertex referenced by this meshlet must lie inside its bounding sphere.
        const glm::vec3 center(m.boundingSphere);
        const float radius = m.boundingSphere.w;
        EXPECT_GE(radius, 0.0f);
        for (std::uint32_t lv = 0; lv < m.vertexCount; ++lv) {
            const std::uint32_t globalIdx = data.meshletVertices[m.vertexOffset + lv];
            ASSERT_LT(globalIdx, mesh.vertices.size());
            const float dist = glm::distance(mesh.vertices[globalIdx].position, center);
            EXPECT_LE(dist, radius + 1e-4f);
        }

        // Reconstruct every global triangle from local uint8 indices + vertex indirection.
        for (std::uint32_t t = 0; t < m.triangleCount; ++t) {
            const std::size_t base = static_cast<std::size_t>(m.triangleOffset) + static_cast<std::size_t>(t) * 3u;
            const std::uint8_t l0 = data.meshletTriangles[base + 0];
            const std::uint8_t l1 = data.meshletTriangles[base + 1];
            const std::uint8_t l2 = data.meshletTriangles[base + 2];
            ASSERT_LT(l0, m.vertexCount);
            ASSERT_LT(l1, m.vertexCount);
            ASSERT_LT(l2, m.vertexCount);

            const std::uint32_t g0 = data.meshletVertices[m.vertexOffset + l0];
            const std::uint32_t g1 = data.meshletVertices[m.vertexOffset + l1];
            const std::uint32_t g2 = data.meshletVertices[m.vertexOffset + l2];
            reconstructedTriangles.insert(canonicalizeTriangle(g0, g1, g2));
        }

        totalTriangles += m.triangleCount;
    }

    EXPECT_EQ(totalTriangles, mesh.indices.size() / 3u);
    EXPECT_EQ(reconstructedTriangles, originalTriangles);
}

} // namespace

TEST(Meshlets, RejectsEmptyAndMalformedMeshes) {
    EXPECT_TRUE(buildMeshlets(MeshData{}).meshlets.empty());

    MeshData nonMultiple = makePrimitive(MeshPrimitive::Cube);
    nonMultiple.indices.pop_back();
    EXPECT_TRUE(buildMeshlets(nonMultiple).meshlets.empty());

    MeshData outOfBounds = makePrimitive(MeshPrimitive::Cube);
    outOfBounds.indices[0] = static_cast<std::uint32_t>(outOfBounds.vertices.size());
    EXPECT_TRUE(buildMeshlets(outOfBounds).meshlets.empty());

    const MeshData valid = makePrimitive(MeshPrimitive::Cube);
    EXPECT_TRUE(buildMeshlets(valid, 0, 64).meshlets.empty());
    EXPECT_TRUE(buildMeshlets(valid, 64, 0).meshlets.empty());
}

TEST(Meshlets, CubeAndSpherePreserveTrianglesAndBounds) {
    const MeshData cube = makePrimitive(MeshPrimitive::Cube);
    const MeshletData cubeMeshlets = buildMeshlets(cube);
    verifyMeshletInvariants(cube, cubeMeshlets, 64, 124);

    const MeshData sphere = makePrimitive(MeshPrimitive::Sphere);
    const MeshletData sphereMeshlets = buildMeshlets(sphere);
    verifyMeshletInvariants(sphere, sphereMeshlets, 64, 124);
}

TEST(Meshlets, ClampsLimitsToHardwareMaximums) {
    const MeshData denseSphere = makeUvSphere(1.0f, 64, 32);
    const MeshletData clamped = buildMeshlets(denseSphere, 255, 512, 0.5f);
    verifyMeshletInvariants(denseSphere, clamped, 64, 124);

    const MeshletData exactCap = buildMeshlets(denseSphere, 64, 126, 0.5f);
    verifyMeshletInvariants(denseSphere, exactCap, 64, 124);
}

TEST(Meshlets, ConeCullingRejectsBackFacingClustersAndKeepsFrontFacingOnes) {
    const MeshData sphere = makeUvSphere(1.0f, 96, 48);
    const MeshletData data = buildMeshlets(sphere, 64, 124, 0.5f);
    ASSERT_GT(data.meshlets.size(), 20u);

    const glm::vec3 cameraLocalPos(0.0f, 0.0f, 5.0f);
    std::size_t culledCount = 0;
    std::size_t visibleCount = 0;

    for (const Meshlet& m : data.meshlets) {
        const bool culled = cullMeshletCone(m, cameraLocalPos);
        if (culled) {
            ++culledCount;
        } else {
            ++visibleCount;
        }

        // Conservative guarantee: any cluster containing a strictly front-facing triangle must stay visible.
        bool hasFrontFacingTri = false;
        for (std::uint32_t t = 0; t < m.triangleCount; ++t) {
            const std::size_t base = static_cast<std::size_t>(m.triangleOffset) + static_cast<std::size_t>(t) * 3u;
            const glm::vec3 p0 =
                sphere.vertices[data.meshletVertices[m.vertexOffset + data.meshletTriangles[base + 0]]].position;
            const glm::vec3 p1 =
                sphere.vertices[data.meshletVertices[m.vertexOffset + data.meshletTriangles[base + 1]]].position;
            const glm::vec3 p2 =
                sphere.vertices[data.meshletVertices[m.vertexOffset + data.meshletTriangles[base + 2]]].position;
            const glm::vec3 triNormal = glm::cross(p1 - p0, p2 - p0);
            const float normalLen = glm::length(triNormal);
            const glm::vec3 toTri = (p0 + p1 + p2) / 3.0f - cameraLocalPos;
            const float toTriLen = glm::length(toTri);
            if (normalLen > 1e-8f && toTriLen > 1e-8f &&
                glm::dot(toTri / toTriLen, triNormal / normalLen) < -1e-3f) {
                hasFrontFacingTri = true;
                break;
            }
        }
        if (hasFrontFacingTri) {
            EXPECT_FALSE(culled) << "Front-facing cluster must never be cone-culled";
        }

        // Clusters squarely on the far side of the unit sphere should be culled.
        if (m.boundingSphere.z < -0.8f && m.coneCutoff < 1.0f) {
            EXPECT_TRUE(culled) << "Far-side sphere cluster should be cone-culled";
        }
    }

    EXPECT_GT(culledCount, data.meshlets.size() / 4u);
    EXPECT_GT(visibleCount, data.meshlets.size() / 4u);

    // Flat plane facing +Y: visible from above, culled from below.
    const MeshData plane = makePlane(2.0f, 16);
    const MeshletData planeData = buildMeshlets(plane);
    ASSERT_FALSE(planeData.meshlets.empty());
    for (const Meshlet& m : planeData.meshlets) {
        EXPECT_FALSE(cullMeshletCone(m, glm::vec3(0.0f, 5.0f, 0.0f)));
        EXPECT_TRUE(cullMeshletCone(m, glm::vec3(0.0f, -5.0f, 0.0f)));
    }
}

TEST(Meshlets, FrustumAndCombinedCullingWorkAsExpected) {
    Meshlet inside{};
    inside.boundingSphere = glm::vec4(0.0f, 0.0f, 0.0f, 0.5f);
    inside.coneApex = glm::vec3(0.0f, 0.0f, 0.0f);
    inside.coneAxis = glm::vec3(0.0f, 0.0f, 1.0f);
    inside.coneCutoff = 0.2f;

    const auto frustum = makeBoxFrustum(glm::vec3(-2.0f), glm::vec3(2.0f));
    EXPECT_FALSE(cullMeshletFrustum(inside, frustum));

    Meshlet outsideLeft = inside;
    outsideLeft.boundingSphere = glm::vec4(-3.0f, 0.0f, 0.0f, 0.5f);
    EXPECT_TRUE(cullMeshletFrustum(outsideLeft, frustum));

    Meshlet touchingBoundary = inside;
    touchingBoundary.boundingSphere = glm::vec4(2.4f, 0.0f, 0.0f, 0.5f);
    EXPECT_FALSE(cullMeshletFrustum(touchingBoundary, frustum));

    // Zeroed unused 6th plane (infinite far plane style) must never falsely cull.
    auto fivePlanes = frustum;
    fivePlanes[5] = glm::vec4(0.0f);
    EXPECT_FALSE(cullMeshletFrustum(inside, fivePlanes));

    // Uncullable cone (cutoff >= 1) and camera-at-apex edge cases never falsely cull or NaN.
    Meshlet uncullable = inside;
    uncullable.coneCutoff = 1.0f;
    EXPECT_FALSE(cullMeshletCone(uncullable, glm::vec3(0.0f, 0.0f, -5.0f)));
    EXPECT_FALSE(cullMeshletCone(inside, inside.coneApex));

    // Combined cullMeshlets over sphere returns strictly visible subset.
    const MeshData sphere = makeUvSphere(1.0f, 64, 32);
    const MeshletData data = buildMeshlets(sphere);
    const glm::vec3 camPos(0.0f, 0.0f, 5.0f);
    const std::vector<std::uint32_t> visible = cullMeshlets(data, frustum, camPos);
    EXPECT_FALSE(visible.empty());
    EXPECT_LT(visible.size(), data.meshlets.size());

    for (std::uint32_t idx : visible) {
        ASSERT_LT(idx, data.meshlets.size());
        EXPECT_FALSE(cullMeshletFrustum(data.meshlets[idx], frustum));
        EXPECT_FALSE(cullMeshletCone(data.meshlets[idx], camPos));
    }
}
