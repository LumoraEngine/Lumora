// Render graph bookkeeping (no GPU). Execution on the GPU is covered by the renderer tests.
#include "renderer/RenderGraph.hpp"
#include "renderer/vulkan/VulkanCommon.hpp"

#include <gtest/gtest.h>

using namespace lumora::render;

TEST(RenderGraph, RecordsPassesAndImagesInOrder) {
    RenderGraph graph;
    const RGImage hdr = graph.createImage("HDR", {nvrhi::Format::RGBA16_FLOAT, 64, 32});
    const RGImage depth = graph.createImage("Depth", {nvrhi::Format::D32, 64, 32});
    graph.addRasterPass(
        "Prepass", [&](PassBuilder& b) { b.writeDepth(depth, LoadOp::Clear, 0.0f); }, [](PassContext&) {});
    graph.addRasterPass(
        "Opaque",
        [&](PassBuilder& b) {
            b.writeColor(hdr, LoadOp::Clear);
            b.readDepth(depth);
            b.recordInParallel();
        },
        [](PassContext&) {});
    graph.addPass("Copy", [&](PassBuilder& b) { b.copyFrom(hdr); }, [](PassContext&) {});
    ASSERT_EQ(graph.passCount(), 3u);
    EXPECT_EQ(graph.passName(0), "Prepass");
    EXPECT_EQ(graph.passName(2), "Copy");
    EXPECT_EQ(graph.imageCount(), 2u);
}

TEST(RenderGraph, RejectsInvalidHandles) {
    RenderGraph graph;
    EXPECT_THROW(graph.addRasterPass(
                     "Bad", [](PassBuilder& b) { b.writeColor(RGImage{}, LoadOp::Clear); }, [](PassContext&) {}),
                 lumora::rhi::VulkanError);
    EXPECT_THROW(graph.addPass("Bad", [](PassBuilder& b) { b.sample(RGImage{42}); }, [](PassContext&) {}),
                 lumora::rhi::VulkanError);
}

TEST(RenderGraph, ComputePassesDeclareStorageImages) {
    RenderGraph graph;
    const RGImage a = graph.createImage("A", {nvrhi::Format::RGBA16_FLOAT, 8, 8, true});
    graph.addComputePass("Fill", [&](PassBuilder& b) { b.writeStorage(a); }, [](PassContext&) {});
    EXPECT_EQ(graph.passCount(), 1u);
    // More storage images than set 2 has slots is refused when the pass is declared.
    std::vector<RGImage> many;
    for (std::uint32_t i = 0; i <= RenderGraph::kStorageSlots; ++i) many.push_back(graph.createImage("I", {nvrhi::Format::R8_UNORM, 4, 4, true}));
    EXPECT_THROW(graph.addComputePass(
                     "TooMany", [&](PassBuilder& b) { for (RGImage i : many) b.writeStorage(i); }, [](PassContext&) {}),
                 lumora::rhi::VulkanError);
}

TEST(RenderGraph, CullsPassesWhoseImagesNothingReads) {
    RenderGraph graph;
    const RGImage out = graph.importImage("Output", {});
    const RGImage hdr = graph.createImage("HDR", {nvrhi::Format::RGBA16_FLOAT, 64, 32});
    const RGImage depth = graph.createImage("Depth", {nvrhi::Format::D32, 64, 32});
    const RGImage unused = graph.createImage("Unused", {nvrhi::Format::RGBA16_FLOAT, 64, 32});
    const RGImage kept = graph.createImage("Kept", {nvrhi::Format::R8_UNORM, 64, 32});
    const RGImage marks = graph.createImage("Marks", {nvrhi::Format::R32_UINT, 8, 8, true});
    auto none = [](PassContext&) {};
    graph.addRasterPass("Prepass", [&](PassBuilder& b) { b.writeDepth(depth, LoadOp::Clear); }, none);       // 0: depth-tested later
    graph.addRasterPass("Sky", [&](PassBuilder& b) { b.writeColor(hdr, LoadOp::Clear); }, none);              // 1: blended into
    graph.addRasterPass("Opaque", [&](PassBuilder& b) {                                                         // 2
        b.writeColor(hdr, LoadOp::Load);
        b.readDepth(depth);
    }, none);
    graph.addRasterPass("Nobody", [&](PassBuilder& b) {                                                         // 3: culled
        b.writeColor(unused, LoadOp::Clear);
        b.sample(hdr);
    }, none);
    graph.addRasterPass("Effects", [&](PassBuilder& b) {                                                        // 4: keeps itself
        b.writeColor(kept, LoadOp::Clear);
        b.keepAlways();
    }, none);
    graph.addComputePass("Mark", [&](PassBuilder& b) { b.writeStorage(marks); }, none);                        // 5: read back
    graph.addPass("Readback", [&](PassBuilder& b) { b.copyFrom(marks); }, none);                               // 6: writes no image
    graph.addRasterPass("Tonemap", [&](PassBuilder& b) {                                                        // 7: imported
        b.sample(hdr);
        b.writeColor(out, LoadOp::DontCare);
    }, none);
    graph.compile();
    for (std::size_t p : {0u, 1u, 2u, 4u, 5u, 6u, 7u}) EXPECT_FALSE(graph.culled(p)) << graph.passName(p);
    EXPECT_TRUE(graph.culled(3));
    EXPECT_EQ(graph.culledCount(), 1u);
    // An image only the culled pass used gets no texture.
    EXPECT_EQ(graph.textureOf(unused), UINT32_MAX);
    EXPECT_NE(graph.textureOf(hdr), UINT32_MAX);
}

TEST(RenderGraph, ImagesApartInTimeShareTextures) {
    RenderGraph graph;
    const RGImage out = graph.importImage("Output", {});
    auto hdr = [&](const char* name, bool storage = false) {
        return graph.createImage(name, {nvrhi::Format::RGBA16_FLOAT, 64, 32, storage});
    };
    const RGImage scene = hdr("Scene");
    const RGImage blurA = hdr("BlurA");
    const RGImage blurB = hdr("BlurB", true);  // storage: its texture is made for storage
    const RGImage half = graph.createImage("Half", {nvrhi::Format::RGBA16_FLOAT, 32, 16});
    const RGImage final = hdr("Final");
    auto none = [](PassContext&) {};
    graph.addRasterPass("Scene", [&](PassBuilder& b) { b.writeColor(scene, LoadOp::Clear); }, none);  // 0
    graph.addRasterPass("BlurA", [&](PassBuilder& b) {                                                  // 1: scene ends
        b.sample(scene);
        b.writeColor(blurA, LoadOp::DontCare);
    }, none);
    graph.addComputePass("BlurB", [&](PassBuilder& b) {                                                 // 2: blurA ends
        b.sample(blurA);
        b.writeStorage(blurB);
    }, none);
    graph.addRasterPass("Half", [&](PassBuilder& b) {                                                   // 3
        b.sample(blurB);
        b.writeColor(half, LoadOp::DontCare);
    }, none);
    graph.addRasterPass("Final", [&](PassBuilder& b) {                                                  // 4: blurB, half end
        b.sample(blurB);
        b.sample(half);
        b.writeColor(final, LoadOp::DontCare);
    }, none);
    graph.addRasterPass("Out", [&](PassBuilder& b) {
        b.sample(final);
        b.writeColor(out, LoadOp::DontCare);
    }, none);
    graph.compile();
    // Scene (0-1) and BlurB (2-4) never live at once; nor do BlurA (1-2) and Final (4-5).
    // Images alive together never share, and Half differs in size.
    EXPECT_EQ(graph.textureOf(scene), graph.textureOf(blurB));
    EXPECT_EQ(graph.textureOf(blurA), graph.textureOf(final));
    EXPECT_NE(graph.textureOf(scene), graph.textureOf(blurA));
    EXPECT_NE(graph.textureOf(half), graph.textureOf(scene));
    EXPECT_NE(graph.textureOf(half), graph.textureOf(blurA));
    const RenderGraph::TransientStats shared = graph.transientStats();
    EXPECT_EQ(shared.images, 5u);
    EXPECT_EQ(shared.textures, 3u);
    const std::uint64_t full = 64ull * 32 * 8;
    EXPECT_EQ(shared.unsharedBytes, 4 * full + full / 4);
    EXPECT_EQ(shared.bytes, 2 * full + full / 4);

    graph.compile(false);  // a texture each
    EXPECT_EQ(graph.transientStats().textures, 5u);
    EXPECT_EQ(graph.transientStats().bytes, shared.unsharedBytes);
}
