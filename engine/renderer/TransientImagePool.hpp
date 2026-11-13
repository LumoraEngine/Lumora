#pragma once

#include "renderer/Gpu.hpp"

#include <memory>
#include <vector>

namespace lumora::render {

class BindlessTable;

/// Bytes of an uncompressed or block-compressed image of that size (one mip level).
[[nodiscard]] std::uint64_t imageBytes(nvrhi::Format format, std::uint32_t width, std::uint32_t height, std::uint32_t depth = 1);

struct PooledImage {
    nvrhi::TextureHandle texture;
    std::uint32_t bindlessIndex = UINT32_MAX;
    std::uint32_t idleFrames = 0;
    bool inUse = false;
};

/// Recycles render-graph transient textures (render targets, depth) from frame to frame. Frames
/// in flight share them: they run one after another on the GPU's queue and every pass moves its
/// images into the state it needs (a barrier after the previous frame's use). A texture is freed
/// only after kMaxIdleFrames frames without use, so no frame in flight still has it.
/// Every texture is registered in the bindless table once, so passes can sample it.
class TransientImagePool {
public:
    static constexpr std::uint32_t kMaxIdleFrames = 8;

    TransientImagePool(nvrhi::IDevice* device, BindlessTable& bindless);
    ~TransientImagePool();
    TransientImagePool(const TransientImagePool&) = delete;
    TransientImagePool& operator=(const TransientImagePool&) = delete;

    /// Start of a frame (after the slot's frame completed): frees textures idle for too long.
    void beginFrame();
    /// depth > 1 makes a 3D texture (storage, not a render target).
    const PooledImage& acquire(nvrhi::Format format, std::uint32_t width, std::uint32_t height, const char* name,
                               bool storage = false, std::uint32_t depth = 1);
    void clear();
    [[nodiscard]] std::size_t size() const noexcept { return images_.size(); }
    /// Memory of every texture held (idle ones too), from their sizes and formats.
    [[nodiscard]] std::uint64_t bytes() const;

private:
    nvrhi::IDevice* device_;
    BindlessTable& bindless_;
    std::vector<std::unique_ptr<PooledImage>> images_;
};

} // namespace lumora::render
