#pragma once

#include "core/Error.hpp"
#include "renderer/Gpu.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lumora::render {

/// What a material map holds, which decides how it is stored on the GPU.
enum class TextureKind {
    Color,   // base colour, emissive: sRGB; BC1, or BC3 with alpha
    Data,    // metallic-roughness, occlusion: linear BC1
    Normal,  // tangent-space normals: BC5 (x, y; the shader rebuilds z)
};

/// A texture with its full mip chain as the GPU takes it (block-compressed or RGBA8).
struct TextureMips {
    nvrhi::Format format = nvrhi::Format::UNKNOWN;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::vector<std::uint8_t>> levels;  // level 0 first

    [[nodiscard]] std::size_t bytes(std::uint32_t firstLevel = 0) const;
    [[nodiscard]] std::uint32_t levelWidth(std::uint32_t level) const { return std::max(width >> level, 1u); }
    [[nodiscard]] std::uint32_t levelHeight(std::uint32_t level) const { return std::max(height >> level, 1u); }
};

/// The block format for a map (RGBA8 when the GPU has no BC support).
[[nodiscard]] nvrhi::Format textureFormat(TextureKind kind, bool hasAlpha, bool blockCompression);

/// Compresses one RGBA8 image (rows top to bottom) into BC1, BC3, BC4 (red) or BC5 (red, green)
/// blocks; edges are padded by repeating the last texel.
[[nodiscard]] std::vector<std::uint8_t> compressBlocks(std::span<const std::uint8_t> rgba, std::uint32_t width,
                                                       std::uint32_t height, nvrhi::Format format);

/// Mip chain (alpha-coverage preserving where there is transparency) in the kind's format.
[[nodiscard]] TextureMips prepareTexture(std::span<const std::uint8_t> rgba, std::uint32_t width, std::uint32_t height,
                                         TextureKind kind, bool blockCompression);

/// The .ltex cache file: "LTX1", format, width, height, level count, then each level's size and bytes.
[[nodiscard]] Result<> saveTextureMips(const std::filesystem::path& file, const TextureMips& mips);
[[nodiscard]] Result<TextureMips> loadTextureMips(const std::filesystem::path& file);

/// Raised whenever prepareTexture() gives different results: old .ltex files are ignored.
inline constexpr std::uint32_t kTexturePreparationVersion = 1;

/// "|color", "|data" or "|normal": appended to texture keys in cache identities.
[[nodiscard]] const char* textureKindSuffix(TextureKind kind);

/// The editor's cache file for a texture: <root>/Saved/TextureCache/<hash>.ltex, from its
/// identity (absolute canonical key + kind suffix) and the source file's size and time.
/// Empty when `root` is empty or the source cannot be read.
[[nodiscard]] std::filesystem::path textureCacheFile(const std::filesystem::path& root, std::string_view id,
                                                     const std::filesystem::path& source);

/// The project-relative form ("Content/T.png", "Content/M.glb#tex2") of a canonical texture key
/// (AssetLibrary::canonicalTextureKey); empty when it lies outside `root`.
[[nodiscard]] std::string relativeTextureKey(const std::filesystem::path& root, std::string_view canonical);

/// A cooked texture of a shipped game: <root>/Cooked/Textures/<hash>.ltex, from the
/// project-relative key, the kind and the source's size (not its time: copies keep the
/// contents, not the time). Written by cookTextures(), read by TextureCache.
[[nodiscard]] std::filesystem::path cookedTextureFile(const std::filesystem::path& root, std::string_view relativeKey,
                                                      TextureKind kind, std::uintmax_t sourceSize);

} // namespace lumora::render
