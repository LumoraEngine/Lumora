#pragma once

#include "core/Error.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace lumora::app {

/// "linux-x64" or "windows-x64": the platform this editor runs on (release package key).
[[nodiscard]] std::string hostPlatform();
/// The player inside a runtime for a platform: "bin/lumora-editor" or "bin/lumora-editor.exe".
[[nodiscard]] std::filesystem::path playerExecutable(std::string_view platform);

/// A platform a game can be exported for, and the runtime the export copies.
struct ExportTarget {
    std::string platform;
    std::filesystem::path runtimeRoot;
};
/// The editor's own platform, plus every build-support package installed next to it with the Hub
/// (<editor>/platforms/<platform>/, the editor package of that platform). `editorRoot` empty =
/// the running executable's installation.
[[nodiscard]] std::vector<ExportTarget> exportTargets(const std::filesystem::path& editorRoot = {});

/// Writes a standalone copy of a game: a runtime (bin/, lib/ and plugins/, i.e. the player,
/// engine modules, shaders and Qt), the project (without Saved/, except the built game library),
/// its textures cooked (block-compressed with their mips, so the game does not compress them on
/// the player's first run) and a launcher:
///
///   <out>/run.sh or <Name>.bat        starts the player full of the game
///   <out>/bin, lib, plugins           runtime
///   <out>/Game/                       the project
///   <out>/Game/Cooked/Textures/       cooked textures (render::cookTextures)
///
/// The runtime and launcher are those of the target platform, so a Linux editor can export a
/// Windows game and back. C++ behaviours are compiled for the editor's platform: exporting for
/// another one leaves the game library out (a warning says so); Lua and visual scripts run
/// everywhere. The output directory must be empty or not exist. Returns the launcher's path.
struct GameExport {
    std::filesystem::path projectRoot;
    std::string name;                 // the game's name (launcher file on Windows)
    std::filesystem::path output;
    /// Where the runtime comes from; empty = the running executable's installation.
    std::filesystem::path runtimeRoot;
    /// Target platform; empty = the host's. The runtime must be one for that platform.
    std::string platform;
    /// Receives what the export had to leave out.
    std::vector<std::string>* warnings = nullptr;
    /// Texture cooking progress (see render::cookTextures); returning false cancels the export.
    std::function<bool(std::size_t done, std::size_t total, const std::string& key)> progress;
};
[[nodiscard]] Result<std::filesystem::path> exportGame(const GameExport& request);

} // namespace lumora::app
