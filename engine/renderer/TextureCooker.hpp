#pragma once

#include "core/Error.hpp"
#include "renderer/TextureCompression.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace lumora::render {

/// A texture a project uses, and what for (which decides its block format).
struct TextureUse {
    std::string key;  // project-relative texture key ("Content/T.png", "Content/M.glb#tex2")
    TextureKind kind = TextureKind::Color;
    friend bool operator==(const TextureUse&, const TextureUse&) = default;
};

/// Every texture a project's content uses:
/// - maps named in scenes, prefabs, materials, UI documents and visual scripts, by the field
///   that names them (normal maps, data maps such as metallic-roughness and occlusion, colour);
/// - the textures of material graphs, with the kind of their node;
/// - the material maps of every model file (scripts may spawn models no scene names);
/// - image files that nothing names, as colour (pictures scripts put on the HUD). Images a
///   field names for something else (terrain heightmaps) are left alone.
/// Sorted by key; Saved/, Cooked/ and .git/ are not searched.
[[nodiscard]] std::vector<TextureUse> collectTextureUses(const std::filesystem::path& projectRoot);

struct TextureCooking {
    std::filesystem::path projectRoot;  // sources (and the editor's Saved/TextureCache) are read here
    std::filesystem::path outputRoot;   // <outputRoot>/Cooked/Textures/ is written
    /// Called before each texture; returning false cancels the cooking.
    std::function<bool(std::size_t done, std::size_t total, const std::string& key)> progress;
    /// Receives textures that could not be cooked (they are prepared when the game loads them).
    std::vector<std::string>* warnings = nullptr;
};

struct CookStats {
    std::size_t cooked = 0;  // compressed now
    std::size_t reused = 0;  // copied from the editor's texture cache
    std::size_t failed = 0;
    std::size_t bytes = 0;   // written in all
};

/// Writes the block-compressed mip chain of every texture the project uses (see
/// collectTextureUses) to <outputRoot>/Cooked/Textures/, named by cookedTextureFile(), so a
/// shipped game uploads textures at once instead of compressing them on the player's first
/// run. The sources stay where they are: CPU users (heightmaps, HUD fallbacks) still read them.
[[nodiscard]] Result<CookStats> cookTextures(const TextureCooking& request);

} // namespace lumora::render
