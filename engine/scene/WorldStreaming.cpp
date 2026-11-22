#include "scene/WorldStreaming.hpp"

#include "core/FileSystem.hpp"
#include "core/Log.hpp"
#include "scene/Components.hpp"
#include "scene/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <system_error>
#include <utility>

namespace lumora {

namespace {

constexpr std::array<std::byte, 4> kLsecMagic{
    std::byte{'L'}, std::byte{'S'}, std::byte{'E'}, std::byte{'C'}};
constexpr std::uint32_t kLsecVersion = 1u;
constexpr std::uint32_t kFlagCompressed = 1u << 0;
constexpr std::size_t kHeaderSize = 28u;
constexpr std::size_t kMaxPayloadBytes = 256u * 1024u * 1024u;
// Minimum serialized size per SectorEntityRecord:
// id(8) + nameLen(4) + pos(12) + rot(12) + scale(12) + meshLen(4) + matLen(4) + bMin(12) + bMax(12) + minLod(1)
constexpr std::size_t kMinEntityRecordBytes = 81u;

constexpr std::size_t kLzMinMatch = 4u;
constexpr std::size_t kLzMaxOffset = 65535u;
constexpr std::uint32_t kLzHashBits = 14u;
constexpr std::size_t kLzHashSize = std::size_t{1} << kLzHashBits;
constexpr std::uint32_t kLzInvalidPos = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] std::uint32_t loadU32Unaligned(const std::byte* ptr) noexcept {
    std::uint32_t value = 0;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

void pushVarInt(std::vector<std::byte>& out, std::size_t value) {
    while (value >= 255u) {
        out.push_back(std::byte{255});
        value -= 255u;
    }
    out.push_back(static_cast<std::byte>(value));
}

void writeU8(std::vector<std::byte>& out, std::uint8_t v) {
    out.push_back(static_cast<std::byte>(v));
}

void writeU16Le(std::vector<std::byte>& out, std::uint16_t v) {
    out.push_back(static_cast<std::byte>(v & 0xFFu));
    out.push_back(static_cast<std::byte>((v >> 8) & 0xFFu));
}

void writeU32Le(std::vector<std::byte>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFu));
    }
}

void writeI32Le(std::vector<std::byte>& out, std::int32_t v) {
    writeU32Le(out, static_cast<std::uint32_t>(v));
}

void writeU64Le(std::vector<std::byte>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFu));
    }
}

void writeF32Le(std::vector<std::byte>& out, float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    writeU32Le(out, bits);
}

void writeVec3Le(std::vector<std::byte>& out, const glm::vec3& v) {
    writeF32Le(out, v.x);
    writeF32Le(out, v.y);
    writeF32Le(out, v.z);
}

Result<> writeStringLe(std::vector<std::byte>& out, std::string_view s) {
    if (s.size() > std::numeric_limits<std::uint32_t>::max()) {
        return makeError("string exceeds 32-bit length limit");
    }
    writeU32Le(out, static_cast<std::uint32_t>(s.size()));
    const auto* raw = reinterpret_cast<const std::byte*>(s.data());
    out.insert(out.end(), raw, raw + s.size());
    return {};
}

class ByteReader {
public:
    explicit ByteReader(std::span<const std::byte> data) noexcept : data_(data) {}

    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - pos_; }
    [[nodiscard]] std::size_t position() const noexcept { return pos_; }

    [[nodiscard]] Result<std::uint8_t> readU8() {
        if (remaining() < 1u) {
            return makeError("unexpected end of .lsector stream reading u8 at offset {}", pos_);
        }
        return static_cast<std::uint8_t>(data_[pos_++]);
    }

    [[nodiscard]] Result<std::uint32_t> readU32Le() {
        if (remaining() < 4u) {
            return makeError("unexpected end of .lsector stream reading u32 at offset {}", pos_);
        }
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(data_[pos_ + static_cast<std::size_t>(i)]) << (i * 8);
        }
        pos_ += 4u;
        return v;
    }

    [[nodiscard]] Result<std::int32_t> readI32Le() {
        auto u = readU32Le();
        if (!u) return std::unexpected(u.error());
        return static_cast<std::int32_t>(*u);
    }

    [[nodiscard]] Result<std::uint64_t> readU64Le() {
        if (remaining() < 8u) {
            return makeError("unexpected end of .lsector stream reading u64 at offset {}", pos_);
        }
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<std::uint64_t>(data_[pos_ + static_cast<std::size_t>(i)]) << (i * 8);
        }
        pos_ += 8u;
        return v;
    }

    [[nodiscard]] Result<float> readF32Le() {
        auto bits = readU32Le();
        if (!bits) return std::unexpected(bits.error());
        float v = 0.0f;
        std::memcpy(&v, &(*bits), sizeof(v));
        return v;
    }

    [[nodiscard]] Result<glm::vec3> readVec3Le() {
        auto x = readF32Le();
        if (!x) return std::unexpected(x.error());
        auto y = readF32Le();
        if (!y) return std::unexpected(y.error());
        auto z = readF32Le();
        if (!z) return std::unexpected(z.error());
        return glm::vec3{*x, *y, *z};
    }

    [[nodiscard]] Result<std::string> readString() {
        auto len = readU32Le();
        if (!len) return std::unexpected(len.error());
        if (static_cast<std::size_t>(*len) > remaining()) {
            return makeError("string length {} exceeds remaining {} bytes in .lsector payload", *len, remaining());
        }
        std::string s(static_cast<std::size_t>(*len), '\0');
        if (*len > 0u) {
            std::memcpy(s.data(), data_.data() + pos_, static_cast<std::size_t>(*len));
            pos_ += static_cast<std::size_t>(*len);
        }
        return s;
    }

private:
    std::span<const std::byte> data_;
    std::size_t pos_ = 0;
};

[[nodiscard]] bool isEntityVisibleAtLod(HlodLevel sectorLod, HlodLevel entityMinLod) noexcept {
    if (sectorLod == HlodLevel::Unloaded || entityMinLod == HlodLevel::Unloaded) {
        return false;
    }
    return static_cast<std::uint8_t>(sectorLod) <= static_cast<std::uint8_t>(entityMinLod);
}

[[nodiscard]] float maxConfiguredRadius(const StreamingConfig& cfg) noexcept {
    return std::max({0.0f, cfg.lod0Radius, cfg.lod1Radius, cfg.lod2Radius});
}

} // namespace

std::string_view toString(HlodLevel level) noexcept {
    switch (level) {
    case HlodLevel::Lod0Full:   return "Lod0Full";
    case HlodLevel::Lod1Medium: return "Lod1Medium";
    case HlodLevel::Lod2Coarse: return "Lod2Coarse";
    case HlodLevel::Unloaded:   return "Unloaded";
    }
    return "Unknown";
}

SectorCoord worldToSectorCoord(const glm::vec3& worldPos, float cellSize) noexcept {
    if (!std::isfinite(worldPos.x) || !std::isfinite(worldPos.z) || cellSize <= 0.0f || !std::isfinite(cellSize)) {
        return SectorCoord{};
    }
    const double cell = static_cast<double>(cellSize);
    const double fx = std::floor(static_cast<double>(worldPos.x) / cell);
    const double fz = std::floor(static_cast<double>(worldPos.z) / cell);
    const auto clampCoord = [](double v) noexcept -> std::int32_t {
        constexpr double kMin = static_cast<double>(std::numeric_limits<std::int32_t>::min());
        constexpr double kMax = static_cast<double>(std::numeric_limits<std::int32_t>::max());
        if (v <= kMin) return std::numeric_limits<std::int32_t>::min();
        if (v >= kMax) return std::numeric_limits<std::int32_t>::max();
        return static_cast<std::int32_t>(v);
    };
    return SectorCoord{clampCoord(fx), clampCoord(fz)};
}

float distanceToSectorXZ(const glm::vec3& focusPos, SectorCoord coord, float cellSize) noexcept {
    if (cellSize <= 0.0f || !std::isfinite(cellSize) || !std::isfinite(focusPos.x) || !std::isfinite(focusPos.z)) {
        return std::numeric_limits<float>::infinity();
    }
    const double cell = static_cast<double>(cellSize);
    const double minX = static_cast<double>(coord.x) * cell;
    const double maxX = minX + cell;
    const double minZ = static_cast<double>(coord.z) * cell;
    const double maxZ = minZ + cell;

    const double px = static_cast<double>(focusPos.x);
    const double pz = static_cast<double>(focusPos.z);
    const double dx = std::max({minX - px, 0.0, px - maxX});
    const double dz = std::max({minZ - pz, 0.0, pz - maxZ});
    return static_cast<float>(std::sqrt(dx * dx + dz * dz));
}

HlodLevel evaluateSectorHlod(float distance, HlodLevel currentLod, const StreamingConfig& config) noexcept {
    if (!std::isfinite(distance) || distance < 0.0f) {
        return HlodLevel::Unloaded;
    }
    const float h = std::max(config.hysteresis, 0.0f);
    const float r0 = std::max(config.lod0Radius, 0.0f);
    const float r1 = std::max(config.lod1Radius, r0);
    const float r2 = std::max(config.lod2Radius, r1);

    if (distance <= r0) {
        return HlodLevel::Lod0Full;
    }
    if (currentLod == HlodLevel::Lod0Full && distance <= r0 + h) {
        return HlodLevel::Lod0Full;
    }
    if (distance <= r1) {
        return HlodLevel::Lod1Medium;
    }
    if ((currentLod == HlodLevel::Lod0Full || currentLod == HlodLevel::Lod1Medium) && distance <= r1 + h) {
        return HlodLevel::Lod1Medium;
    }
    if (distance <= r2) {
        return HlodLevel::Lod2Coarse;
    }
    if (currentLod != HlodLevel::Unloaded && distance <= r2 + h) {
        return HlodLevel::Lod2Coarse;
    }
    return HlodLevel::Unloaded;
}

std::filesystem::path sectorFilePath(const std::filesystem::path& sectorRoot, SectorCoord coord) {
    return sectorRoot / std::format("sector_{}_{}.lsector", coord.x, coord.z);
}

std::uint64_t fnv1a64(std::span<const std::byte> data) noexcept {
    constexpr std::uint64_t kOffsetBasis = 14695981039346656037ull;
    constexpr std::uint64_t kPrime = 1099511628211ull;
    std::uint64_t hash = kOffsetBasis;
    for (const std::byte b : data) {
        hash ^= static_cast<std::uint8_t>(b);
        hash *= kPrime;
    }
    return hash;
}

std::vector<std::byte> compressSectorBlock(std::span<const std::byte> input) {
    std::vector<std::byte> out;
    if (input.empty()) {
        return out;
    }

    if (input.size() < 12u) {
        out.reserve(input.size() + 2u);
        const std::size_t litLen = input.size();
        const std::uint8_t litNibble = static_cast<std::uint8_t>(std::min<std::size_t>(litLen, 15u));
        writeU8(out, static_cast<std::uint8_t>(litNibble << 4));
        if (litLen >= 15u) {
            pushVarInt(out, litLen - 15u);
        }
        out.insert(out.end(), input.begin(), input.end());
        return out;
    }

    out.reserve(input.size());
    std::vector<std::uint32_t> hashTable(kLzHashSize, kLzInvalidPos);

    const std::size_t limit = input.size() - kLzMinMatch;
    std::size_t ip = 0;
    std::size_t anchor = 0;

    while (ip <= limit) {
        const std::uint32_t seq = loadU32Unaligned(input.data() + ip);
        const std::uint32_t slot = ((seq * 2654435761u) >> (32u - kLzHashBits)) & (kLzHashSize - 1u);
        const std::uint32_t candidate = hashTable[slot];
        hashTable[slot] = static_cast<std::uint32_t>(ip);

        if (candidate != kLzInvalidPos && ip > candidate && (ip - candidate) <= kLzMaxOffset &&
            loadU32Unaligned(input.data() + candidate) == seq) {
            std::size_t matchLen = kLzMinMatch;
            while (ip + matchLen < input.size() && input[candidate + matchLen] == input[ip + matchLen]) {
                ++matchLen;
            }

            const std::size_t litLen = ip - anchor;
            const std::size_t matchExtra = matchLen - kLzMinMatch;
            const std::uint8_t litNibble = static_cast<std::uint8_t>(std::min<std::size_t>(litLen, 15u));
            const std::uint8_t matchNibble = static_cast<std::uint8_t>(std::min<std::size_t>(matchExtra, 15u));
            writeU8(out, static_cast<std::uint8_t>((litNibble << 4) | matchNibble));

            if (litLen >= 15u) {
                pushVarInt(out, litLen - 15u);
            }
            out.insert(out.end(), input.begin() + static_cast<std::ptrdiff_t>(anchor),
                       input.begin() + static_cast<std::ptrdiff_t>(ip));

            const std::uint16_t offset = static_cast<std::uint16_t>(ip - candidate);
            writeU16Le(out, offset);

            if (matchExtra >= 15u) {
                pushVarInt(out, matchExtra - 15u);
            }

            ip += matchLen;
            anchor = ip;
            if (ip > 0u && ip - 1u <= limit) {
                const std::uint32_t tailSeq = loadU32Unaligned(input.data() + ip - 1u);
                const std::uint32_t tailSlot = ((tailSeq * 2654435761u) >> (32u - kLzHashBits)) & (kLzHashSize - 1u);
                hashTable[tailSlot] = static_cast<std::uint32_t>(ip - 1u);
            }
        } else {
            ++ip;
        }
    }

    if (anchor < input.size()) {
        const std::size_t litLen = input.size() - anchor;
        const std::uint8_t litNibble = static_cast<std::uint8_t>(std::min<std::size_t>(litLen, 15u));
        writeU8(out, static_cast<std::uint8_t>(litNibble << 4));
        if (litLen >= 15u) {
            pushVarInt(out, litLen - 15u);
        }
        out.insert(out.end(), input.begin() + static_cast<std::ptrdiff_t>(anchor), input.end());
    }

    return out;
}

Result<std::vector<std::byte>> decompressSectorBlock(std::span<const std::byte> compressed, std::size_t expectedSize) {
    if (expectedSize > kMaxPayloadBytes) {
        return makeError("uncompressed sector size {} exceeds maximum {}", expectedSize, kMaxPayloadBytes);
    }
    if (expectedSize == 0u) {
        if (!compressed.empty()) {
            return makeError("non-empty compressed stream for zero-byte sector payload");
        }
        return std::vector<std::byte>{};
    }
    if (compressed.empty()) {
        return makeError("empty compressed stream for {}-byte sector payload", expectedSize);
    }

    std::vector<std::byte> out;
    out.reserve(expectedSize);

    std::size_t inPos = 0;
    while (inPos < compressed.size()) {
        const std::uint8_t token = static_cast<std::uint8_t>(compressed[inPos++]);
        std::size_t litLen = static_cast<std::size_t>(token >> 4);
        if (litLen == 15u) {
            while (true) {
                if (inPos >= compressed.size()) {
                    return makeError("truncated LZ literal length in .lsector payload");
                }
                const std::uint8_t extra = static_cast<std::uint8_t>(compressed[inPos++]);
                const std::size_t avail = expectedSize - out.size();
                if (litLen > avail || static_cast<std::size_t>(extra) > avail - litLen) {
                    return makeError("LZ literal length exceeds expected uncompressed size");
                }
                litLen += static_cast<std::size_t>(extra);
                if (extra < 255u) {
                    break;
                }
            }
        }

        if (litLen > compressed.size() - inPos) {
            return makeError("truncated LZ literal bytes in .lsector payload");
        }
        if (litLen > expectedSize - out.size()) {
            return makeError("LZ literal output overflow in .lsector payload");
        }
        if (litLen > 0u) {
            out.insert(out.end(), compressed.begin() + static_cast<std::ptrdiff_t>(inPos),
                       compressed.begin() + static_cast<std::ptrdiff_t>(inPos + litLen));
            inPos += litLen;
        }

        if (inPos == compressed.size()) {
            break;
        }

        if (compressed.size() - inPos < 2u) {
            return makeError("truncated LZ match offset in .lsector payload");
        }
        const std::uint16_t offset = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(compressed[inPos]) |
            (static_cast<std::uint16_t>(compressed[inPos + 1u]) << 8));
        inPos += 2u;

        if (offset == 0u || static_cast<std::size_t>(offset) > out.size()) {
            return makeError("invalid LZ match offset {} at output position {}", offset, out.size());
        }

        std::size_t matchLen = static_cast<std::size_t>(token & 0x0Fu) + kLzMinMatch;
        if ((token & 0x0Fu) == 15u) {
            while (true) {
                if (inPos >= compressed.size()) {
                    return makeError("truncated LZ match length in .lsector payload");
                }
                const std::uint8_t extra = static_cast<std::uint8_t>(compressed[inPos++]);
                const std::size_t avail = expectedSize - out.size();
                if (matchLen > avail || static_cast<std::size_t>(extra) > avail - matchLen) {
                    return makeError("LZ match length exceeds expected uncompressed size");
                }
                matchLen += static_cast<std::size_t>(extra);
                if (extra < 255u) {
                    break;
                }
            }
        }

        if (matchLen > expectedSize - out.size()) {
            return makeError("LZ match output overflow in .lsector payload");
        }

        const std::size_t oldSize = out.size();
        out.resize(oldSize + matchLen);
        const std::size_t srcBase = oldSize - static_cast<std::size_t>(offset);
        for (std::size_t i = 0; i < matchLen; ++i) {
            out[oldSize + i] = out[srcBase + i];
        }
    }

    if (out.size() != expectedSize) {
        return makeError("LZ decompressed size mismatch: expected {}, got {}", expectedSize, out.size());
    }
    return out;
}

Result<std::vector<std::byte>> encodeSectorPackage(const SectorPackage& pkg) {
    if (pkg.entities.size() > std::numeric_limits<std::uint32_t>::max()) {
        return makeError("too many entities in sector ({}, {})", pkg.coord.x, pkg.coord.z);
    }

    std::vector<std::byte> payload;
    payload.reserve(32u + pkg.entities.size() * 112u);

    writeI32Le(payload, pkg.coord.x);
    writeI32Le(payload, pkg.coord.z);
    writeVec3Le(payload, pkg.boundsMin);
    writeVec3Le(payload, pkg.boundsMax);
    writeU32Le(payload, static_cast<std::uint32_t>(pkg.entities.size()));

    for (const SectorEntityRecord& rec : pkg.entities) {
        writeU64Le(payload, rec.id.value());
        if (auto r = writeStringLe(payload, rec.name); !r) {
            return std::unexpected(r.error());
        }
        writeVec3Le(payload, rec.position);
        writeVec3Le(payload, rec.rotationDeg);
        writeVec3Le(payload, rec.scale);
        if (auto r = writeStringLe(payload, rec.meshAsset); !r) {
            return std::unexpected(r.error());
        }
        if (auto r = writeStringLe(payload, rec.materialAsset); !r) {
            return std::unexpected(r.error());
        }
        writeVec3Le(payload, rec.boundsMin);
        writeVec3Le(payload, rec.boundsMax);
        writeU8(payload, static_cast<std::uint8_t>(rec.minLod));
    }

    if (payload.size() > kMaxPayloadBytes) {
        return makeError("sector payload size {} exceeds maximum {}", payload.size(), kMaxPayloadBytes);
    }

    const std::uint64_t checksum = fnv1a64(payload);
    const std::vector<std::byte> compressed = compressSectorBlock(payload);
    const bool useCompressed = !compressed.empty() && compressed.size() < payload.size();
    const std::span<const std::byte> storedSpan = useCompressed ? std::span<const std::byte>(compressed)
                                                                : std::span<const std::byte>(payload);

    std::vector<std::byte> fileBytes;
    fileBytes.reserve(kHeaderSize + storedSpan.size());
    fileBytes.insert(fileBytes.end(), kLsecMagic.begin(), kLsecMagic.end());
    writeU32Le(fileBytes, kLsecVersion);
    writeU32Le(fileBytes, useCompressed ? kFlagCompressed : 0u);
    writeU32Le(fileBytes, static_cast<std::uint32_t>(payload.size()));
    writeU32Le(fileBytes, static_cast<std::uint32_t>(storedSpan.size()));
    writeU64Le(fileBytes, checksum);
    fileBytes.insert(fileBytes.end(), storedSpan.begin(), storedSpan.end());
    return fileBytes;
}

Result<SectorPackage> decodeSectorPackage(std::span<const std::byte> bytes) {
    if (bytes.size() < kHeaderSize) {
        return makeError("invalid .lsector file: size {} smaller than {}-byte header", bytes.size(), kHeaderSize);
    }
    if (!std::equal(kLsecMagic.begin(), kLsecMagic.end(), bytes.begin())) {
        return makeError("invalid .lsector magic header");
    }

    ByteReader hdr(bytes.subspan(4u, kHeaderSize - 4u));
    auto version = hdr.readU32Le();
    auto flags = hdr.readU32Le();
    auto uncompressedSize = hdr.readU32Le();
    auto storedSize = hdr.readU32Le();
    auto expectedChecksum = hdr.readU64Le();
    if (!version || !flags || !uncompressedSize || !storedSize || !expectedChecksum) {
        return makeError("corrupt .lsector header");
    }
    if (*version != kLsecVersion) {
        return makeError("unsupported .lsector version {}", *version);
    }
    if ((*flags & ~kFlagCompressed) != 0u) {
        return makeError("unknown .lsector flags {:#x}", *flags);
    }
    if (static_cast<std::size_t>(*uncompressedSize) > kMaxPayloadBytes) {
        return makeError(".lsector uncompressed size {} exceeds limit {}", *uncompressedSize, kMaxPayloadBytes);
    }
    if (static_cast<std::size_t>(*storedSize) != bytes.size() - kHeaderSize) {
        return makeError(".lsector payload size mismatch: header={}, actual={}", *storedSize, bytes.size() - kHeaderSize);
    }

    const std::span<const std::byte> storedPayload = bytes.subspan(kHeaderSize);
    std::vector<std::byte> rawStorage;
    std::span<const std::byte> uncompressedSpan;

    if ((*flags & kFlagCompressed) != 0u) {
        auto decompressed = decompressSectorBlock(storedPayload, static_cast<std::size_t>(*uncompressedSize));
        if (!decompressed) {
            return std::unexpected(decompressed.error());
        }
        rawStorage = std::move(*decompressed);
        uncompressedSpan = rawStorage;
    } else {
        if (*storedSize != *uncompressedSize) {
            return makeError(".lsector raw payload size mismatch: stored={}, uncompressed={}", *storedSize,
                             *uncompressedSize);
        }
        uncompressedSpan = storedPayload;
    }

    const std::uint64_t actualChecksum = fnv1a64(uncompressedSpan);
    if (actualChecksum != *expectedChecksum) {
        return makeError(".lsector checksum mismatch: expected {:#018x}, got {:#018x}", *expectedChecksum,
                         actualChecksum);
    }

    ByteReader reader(uncompressedSpan);
    SectorPackage pkg;
    auto cx = reader.readI32Le();
    auto cz = reader.readI32Le();
    auto bMin = reader.readVec3Le();
    auto bMax = reader.readVec3Le();
    auto count = reader.readU32Le();
    if (!cx || !cz || !bMin || !bMax || !count) {
        return makeError("corrupt .lsector sector metadata");
    }

    pkg.coord = SectorCoord{*cx, *cz};
    pkg.boundsMin = *bMin;
    pkg.boundsMax = *bMax;

    if (static_cast<std::size_t>(*count) > reader.remaining() / kMinEntityRecordBytes) {
        return makeError(".lsector entity count {} exceeds payload bounds", *count);
    }

    pkg.entities.reserve(*count);
    for (std::uint32_t i = 0; i < *count; ++i) {
        SectorEntityRecord rec;
        auto rawUuid = reader.readU64Le();
        if (!rawUuid) return std::unexpected(rawUuid.error());
        rec.id = Uuid(*rawUuid);

        auto name = reader.readString();
        if (!name) return std::unexpected(name.error());
        rec.name = std::move(*name);

        auto pos = reader.readVec3Le();
        if (!pos) return std::unexpected(pos.error());
        rec.position = *pos;

        auto rot = reader.readVec3Le();
        if (!rot) return std::unexpected(rot.error());
        rec.rotationDeg = *rot;

        auto scale = reader.readVec3Le();
        if (!scale) return std::unexpected(scale.error());
        rec.scale = *scale;

        auto mesh = reader.readString();
        if (!mesh) return std::unexpected(mesh.error());
        rec.meshAsset = std::move(*mesh);

        auto mat = reader.readString();
        if (!mat) return std::unexpected(mat.error());
        rec.materialAsset = std::move(*mat);

        auto eMin = reader.readVec3Le();
        if (!eMin) return std::unexpected(eMin.error());
        rec.boundsMin = *eMin;

        auto eMax = reader.readVec3Le();
        if (!eMax) return std::unexpected(eMax.error());
        rec.boundsMax = *eMax;

        auto lodByte = reader.readU8();
        if (!lodByte) return std::unexpected(lodByte.error());
        if (*lodByte > static_cast<std::uint8_t>(HlodLevel::Unloaded)) {
            return makeError(".lsector invalid HlodLevel {} on entity {}", *lodByte, i);
        }
        rec.minLod = static_cast<HlodLevel>(*lodByte);

        pkg.entities.push_back(std::move(rec));
    }

    if (reader.remaining() != 0u) {
        return makeError(".lsector payload has {} unexpected trailing bytes", reader.remaining());
    }

    return pkg;
}

Result<> saveSectorPackage(const std::filesystem::path& file, const SectorPackage& pkg) {
    auto encoded = encodeSectorPackage(pkg);
    if (!encoded) {
        return std::unexpected(encoded.error());
    }
    return fs::writeFileAtomic(file, *encoded);
}

Result<SectorPackage> loadSectorPackage(const std::filesystem::path& file) {
    auto bytes = fs::readBinaryFile(file);
    if (!bytes) {
        return std::unexpected(bytes.error());
    }
    return decodeSectorPackage(*bytes);
}

// =================================================================================================
// WorldStreamer
// =================================================================================================

WorldStreamer::WorldStreamer(std::filesystem::path sectorRoot, StreamingConfig config)
    : sectorRoot_(std::move(sectorRoot)),
      config_(config),
      ioThread_([this](std::stop_token st) { workerLoop(std::move(st)); }) {}

WorldStreamer::~WorldStreamer() { shutdown(); }

void WorldStreamer::shutdown() {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    ioThread_.request_stop();
    queueCv_.notify_all();
    if (ioThread_.joinable()) {
        ioThread_.join();
    }
}

void WorldStreamer::scheduleDirtyLockedMain(SectorCoord coord, ActiveSector& sector) {
    if (!sector.needsReconcile && sector.desiredLod == sector.activeLod) {
        sector.dirtyState = DirtyState::None;
        return;
    }
    if (sector.desiredLod == HlodLevel::Unloaded) {
        if (sector.dirtyState != DirtyState::Unload) {
            sector.dirtyState = DirtyState::Unload;
            dirtyUnloads_.push_back(coord);
        }
    } else {
        if (sector.dirtyState != DirtyState::LodUpdate) {
            sector.dirtyState = DirtyState::LodUpdate;
            dirtyLodUpdates_.push_back(coord);
        }
    }
}

void WorldStreamer::updateFocus(const glm::vec3& focusPos) {
    if (!std::isfinite(focusPos.x) || !std::isfinite(focusPos.z)) {
        return;
    }

    std::scoped_lock mainLock(mainStateMutex_);
    const StreamingConfig cfg = config_;
    if (cfg.cellSize <= 0.0f || !std::isfinite(cfg.cellSize)) {
        return;
    }

    const float cell = cfg.cellSize;
    const float maxRadius = maxConfiguredRadius(cfg);
    const float hysteresisExitRadius = maxRadius + std::max(cfg.hysteresis, 0.0f);

    // 1. Update desired HLOD (or mark unload) on active sectors using hysteresis.
    for (auto& [coord, active] : activeSectors_) {
        const float dist = distanceToSectorXZ(focusPos, coord, cell);
        const HlodLevel nextDesired = evaluateSectorHlod(dist, active.desiredLod, cfg);
        if (nextDesired != active.desiredLod) {
            active.desiredLod = nextDesired;
            active.reconcileCursor = 0u;
            active.needsReconcile = true;
            scheduleDirtyLockedMain(coord, active);
        }
    }

    // 2. Update partial mid-commit sector if one is currently active.
    if (partialCommit_.has_value()) {
        const float dist = distanceToSectorXZ(focusPos, partialCommit_->coord, cell);
        const HlodLevel nextDesired = evaluateSectorHlod(dist, partialCommit_->desiredLod, cfg);
        if (nextDesired != partialCommit_->desiredLod) {
            partialCommit_->desiredLod = nextDesired;
            partialCommit_->reconcileCursor = 0u;
        }
    }

    // 3. Re-evaluate staged, pending/in-flight, and absent sectors under queueMutex_.
    bool enqueuedAny = false;
    {
        std::scoped_lock queueLock(queueMutex_);

        // 3a. Update staged sectors or cancel out-of-range ones before commit.
        for (auto it = stagedQueue_.begin(); it != stagedQueue_.end();) {
            const float dist = distanceToSectorXZ(focusPos, it->coord, cell);
            const HlodLevel nextLod = evaluateSectorHlod(dist, it->targetLod, cfg);
            if (nextLod == HlodLevel::Unloaded) {
                it = stagedQueue_.erase(it);
            } else {
                it->targetLod = nextLod;
                ++it;
            }
        }

        // 3b. Update pending/in-flight requests or cancel out-of-range ones.
        for (auto it = pendingMap_.begin(); it != pendingMap_.end();) {
            const float dist = distanceToSectorXZ(focusPos, it->first, cell);
            const HlodLevel nextLod = evaluateSectorHlod(dist, it->second.targetLod, cfg);
            if (nextLod == HlodLevel::Unloaded) {
                const SectorCoord cancelledCoord = it->first;
                std::erase_if(loadQueue_, [cancelledCoord](const LoadRequest& req) {
                    return req.coord == cancelledCoord;
                });
                it = pendingMap_.erase(it);
            } else {
                it->second.targetLod = nextLod;
                ++it;
            }
        }
        for (LoadRequest& req : loadQueue_) {
            const float d = distanceToSectorXZ(focusPos, req.coord, cell);
            req.distSq = d * d;
            if (const auto pit = pendingMap_.find(req.coord); pit != pendingMap_.end()) {
                req.targetLod = pit->second.targetLod;
            }
        }

        // 3c. Expire absent cache entries once outside the unload hysteresis horizon.
        for (auto it = absentSectors_.begin(); it != absentSectors_.end();) {
            const float dist = distanceToSectorXZ(focusPos, *it, cell);
            if (dist > hysteresisExitRadius) {
                it = absentSectors_.erase(it);
            } else {
                ++it;
            }
        }

        // 4. Scan grid window around focusPos for missing sectors within lod2Radius.
        const double dCell = static_cast<double>(cell);
        const double dRad = static_cast<double>(maxRadius);
        const auto clampInt64ToI32 = [](double v, std::int64_t offset) noexcept -> std::int32_t {
            const double f = std::floor(v);
            constexpr double kMinD = static_cast<double>(std::numeric_limits<std::int32_t>::min());
            constexpr double kMaxD = static_cast<double>(std::numeric_limits<std::int32_t>::max());
            if (f <= kMinD) return std::numeric_limits<std::int32_t>::min();
            if (f >= kMaxD) return std::numeric_limits<std::int32_t>::max();
            const std::int64_t shifted = static_cast<std::int64_t>(f) + offset;
            return static_cast<std::int32_t>(std::clamp<std::int64_t>(
                shifted, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()));
        };

        const std::int32_t minX = clampInt64ToI32((static_cast<double>(focusPos.x) - dRad) / dCell, -1);
        const std::int32_t maxX = clampInt64ToI32((static_cast<double>(focusPos.x) + dRad) / dCell, 1);
        const std::int32_t minZ = clampInt64ToI32((static_cast<double>(focusPos.z) - dRad) / dCell, -1);
        const std::int32_t maxZ = clampInt64ToI32((static_cast<double>(focusPos.z) + dRad) / dCell, 1);

        auto isStaged = [&](SectorCoord c) {
            return std::any_of(stagedQueue_.begin(), stagedQueue_.end(),
                               [c](const StagedSector& s) { return s.coord == c; });
        };

        for (std::int32_t cz = minZ; cz <= maxZ; ++cz) {
            for (std::int32_t cx = minX; cx <= maxX; ++cx) {
                const SectorCoord coord{cx, cz};
                if (activeSectors_.contains(coord)) continue;
                if (partialCommit_.has_value() && partialCommit_->coord == coord) continue;
                if (pendingMap_.contains(coord)) continue;
                if (absentSectors_.contains(coord)) continue;
                if (isStaged(coord)) continue;

                const float dist = distanceToSectorXZ(focusPos, coord, cell);
                const HlodLevel targetLod = evaluateSectorHlod(dist, HlodLevel::Unloaded, cfg);
                if (targetLod == HlodLevel::Unloaded) {
                    continue;
                }

                const std::uint64_t gen = ++nextGeneration_;
                pendingMap_.emplace(coord, PendingState{gen, targetLod});
                loadQueue_.push_back(LoadRequest{coord, targetLod, gen, dist * dist});
                enqueuedAny = true;
            }
            if (cz == maxZ) break; // guard against int32 wrap
        }
    }

    if (enqueuedAny) {
        queueCv_.notify_one();
    }
}

bool WorldStreamer::reconcileSectorStep(Scene& scene, ActiveSector& sector, bool useBudget,
                                        std::chrono::steady_clock::time_point startTime,
                                        std::chrono::duration<double, std::milli> budget, std::size_t& opsDone) {
    auto budgetReached = [&]() -> bool {
        return useBudget && opsDone > 0u && (std::chrono::steady_clock::now() - startTime) >= budget;
    };

    const std::size_t total = sector.package.entities.size();
    while (sector.reconcileCursor < total) {
        if (budgetReached()) {
            return false;
        }
        const std::size_t idx = sector.reconcileCursor;
        const SectorEntityRecord& rec = sector.package.entities[idx];
        const bool wantEntity = isEntityVisibleAtLod(sector.desiredLod, rec.minLod);
        entt::entity& slot = sector.entities[idx];

        if (wantEntity && slot == entt::null) {
            const entt::entity e = scene.createEntity(rec.name, rec.id);
            auto& reg = scene.registry();
            auto& tf = reg.get<TransformComponent>(e);
            tf.position = rec.position;
            tf.rotationDeg = rec.rotationDeg;
            tf.scale = rec.scale;
            reg.get<WorldTransformComponent>(e).matrix = tf.localMatrix();
            reg.emplace<StreamedSectorTag>(e, StreamedSectorTag{sector.coord, rec.minLod});
            if (!rec.meshAsset.empty() || !rec.materialAsset.empty()) {
                auto& mr = reg.emplace<MeshRendererComponent>(e);
                mr.meshAsset = rec.meshAsset;
                mr.material = rec.materialAsset;
            }
            slot = e;
            activeEntitiesCount_.fetch_add(1u, std::memory_order_relaxed);
            ++opsDone;
        } else if (!wantEntity && slot != entt::null) {
            if (scene.isValid(slot)) {
                scene.destroyEntity(slot);
            }
            slot = entt::null;
            activeEntitiesCount_.fetch_sub(1u, std::memory_order_relaxed);
            ++opsDone;
        }
        ++sector.reconcileCursor;
    }

    sector.activeLod = sector.desiredLod;
    sector.needsReconcile = false;
    return true;
}

void WorldStreamer::commitReadySectors(Scene& scene) {
    std::scoped_lock mainLock(mainStateMutex_);

    const auto startTime = std::chrono::steady_clock::now();
    const bool useBudget = config_.commitBudgetMs > 0.0f;
    const auto budget = std::chrono::duration<double, std::milli>(static_cast<double>(config_.commitBudgetMs));
    std::size_t opsDone = 0;

    auto budgetReached = [&]() -> bool {
        return useBudget && opsDone > 0u && (std::chrono::steady_clock::now() - startTime) >= budget;
    };

    // 1. Process pending sector unloads first so destroyed entities free scene slots promptly.
    while (!dirtyUnloads_.empty()) {
        if (budgetReached()) {
            return;
        }
        const SectorCoord coord = dirtyUnloads_.front();
        auto it = activeSectors_.find(coord);
        if (it == activeSectors_.end() || it->second.dirtyState != DirtyState::Unload ||
            it->second.desiredLod != HlodLevel::Unloaded) {
            dirtyUnloads_.pop_front();
            continue;
        }
        ActiveSector& sector = it->second;
        if (!reconcileSectorStep(scene, sector, useBudget, startTime, budget, opsDone)) {
            return;
        }
        // If package had 0 entities, still count the unload step as progress.
        if (sector.package.entities.empty()) {
            ++opsDone;
        }
        dirtyUnloads_.pop_front();
        activeSectors_.erase(it);
        activeSectorsCount_.fetch_sub(1u, std::memory_order_release);
        unloadsCompleted_.fetch_add(1u, std::memory_order_release);
    }

    // 2. Process HLOD upgrades/downgrades on already-active sectors.
    while (!dirtyLodUpdates_.empty()) {
        if (budgetReached()) {
            return;
        }
        const SectorCoord coord = dirtyLodUpdates_.front();
        auto it = activeSectors_.find(coord);
        if (it == activeSectors_.end() || it->second.dirtyState != DirtyState::LodUpdate) {
            dirtyLodUpdates_.pop_front();
            continue;
        }
        ActiveSector& sector = it->second;
        if (!sector.needsReconcile && sector.activeLod == sector.desiredLod) {
            sector.dirtyState = DirtyState::None;
            dirtyLodUpdates_.pop_front();
            continue;
        }
        if (!reconcileSectorStep(scene, sector, useBudget, startTime, budget, opsDone)) {
            return;
        }
        sector.dirtyState = DirtyState::None;
        dirtyLodUpdates_.pop_front();
    }

    // 3. Finish any partially committed sector carried over from the previous frame.
    auto finishPartialCommit = [&]() -> bool {
        if (!partialCommit_.has_value()) {
            return true;
        }
        ActiveSector& sector = *partialCommit_;
        if (!reconcileSectorStep(scene, sector, useBudget, startTime, budget, opsDone)) {
            return false;
        }
        if (sector.package.entities.empty()) {
            ++opsDone;
        }
        const SectorCoord coord = sector.coord;
        if (sector.desiredLod == HlodLevel::Unloaded) {
            partialCommit_.reset();
            partialCommitActive_.store(false, std::memory_order_release);
            return true;
        }
        activeSectors_.insert_or_assign(coord, std::move(sector));
        partialCommit_.reset();
        partialCommitActive_.store(false, std::memory_order_release);
        activeSectorsCount_.fetch_add(1u, std::memory_order_release);
        loadsCompleted_.fetch_add(1u, std::memory_order_release);
        return true;
    };

    if (!finishPartialCommit()) {
        return;
    }

    // 4. Pull newly staged sectors from the background queue and commit within budget.
    while (!budgetReached()) {
        std::optional<StagedSector> nextStaged;
        {
            std::scoped_lock queueLock(queueMutex_);
            if (stagedQueue_.empty()) {
                break;
            }
            nextStaged = std::move(stagedQueue_.front());
            stagedQueue_.pop_front();
        }

        if (nextStaged->targetLod == HlodLevel::Unloaded || activeSectors_.contains(nextStaged->coord)) {
            continue;
        }

        ActiveSector incoming;
        incoming.coord = nextStaged->coord;
        incoming.activeLod = HlodLevel::Unloaded;
        incoming.desiredLod = nextStaged->targetLod;
        incoming.entities.assign(nextStaged->package.entities.size(), entt::null);
        incoming.package = std::move(nextStaged->package);
        incoming.reconcileCursor = 0u;
        incoming.dirtyState = DirtyState::None;

        partialCommit_ = std::move(incoming);
        partialCommitActive_.store(true, std::memory_order_release);
        if (!finishPartialCommit()) {
            return;
        }
    }
}

StreamingStats WorldStreamer::stats() const {
    std::scoped_lock queueLock(queueMutex_);
    StreamingStats s;
    s.loadingSectors = pendingMap_.size();
    s.stagedSectors = stagedQueue_.size() + (partialCommitActive_.load(std::memory_order_acquire) ? 1u : 0u);
    s.activeSectors = activeSectorsCount_.load(std::memory_order_acquire);
    s.activeEntities = activeEntitiesCount_.load(std::memory_order_acquire);
    s.loadsCompleted = loadsCompleted_.load(std::memory_order_acquire);
    s.unloadsCompleted = unloadsCompleted_.load(std::memory_order_acquire);
    return s;
}

HlodLevel WorldStreamer::sectorHlod(SectorCoord coord) const {
    std::scoped_lock mainLock(mainStateMutex_);
    if (const auto it = activeSectors_.find(coord); it != activeSectors_.end()) {
        return it->second.activeLod != HlodLevel::Unloaded ? it->second.activeLod : it->second.desiredLod;
    }
    if (partialCommit_.has_value() && partialCommit_->coord == coord) {
        return partialCommit_->desiredLod;
    }
    std::scoped_lock queueLock(queueMutex_);
    for (const StagedSector& s : stagedQueue_) {
        if (s.coord == coord) {
            return s.targetLod;
        }
    }
    if (const auto pit = pendingMap_.find(coord); pit != pendingMap_.end()) {
        return pit->second.targetLod;
    }
    return HlodLevel::Unloaded;
}

std::vector<entt::entity> WorldStreamer::entitiesInSector(SectorCoord coord) const {
    std::scoped_lock mainLock(mainStateMutex_);
    std::vector<entt::entity> result;
    auto collect = [&](const ActiveSector& sec) {
        result.reserve(sec.entities.size());
        for (entt::entity e : sec.entities) {
            if (e != entt::null) {
                result.push_back(e);
            }
        }
    };
    if (const auto it = activeSectors_.find(coord); it != activeSectors_.end()) {
        collect(it->second);
    } else if (partialCommit_.has_value() && partialCommit_->coord == coord) {
        collect(*partialCommit_);
    }
    return result;
}

void WorldStreamer::invalidateSector(SectorCoord coord) {
    std::scoped_lock queueLock(queueMutex_);
    absentSectors_.erase(coord);
}

void WorldStreamer::setConfig(const StreamingConfig& config) {
    std::scoped_lock mainLock(mainStateMutex_);
    config_ = config;
}

void WorldStreamer::workerLoop(std::stop_token stopToken) {
    while (!stopToken.stop_requested()) {
        LoadRequest req;
        {
            std::unique_lock lock(queueMutex_);
            queueCv_.wait(lock, stopToken, [&]() { return !loadQueue_.empty(); });
            if (stopToken.stop_requested()) {
                return;
            }
            if (loadQueue_.empty()) {
                continue;
            }
            // Pop the request closest to focusPos so high-priority sectors load first.
            auto bestIt = std::min_element(loadQueue_.begin(), loadQueue_.end(),
                                           [](const LoadRequest& a, const LoadRequest& b) {
                                               return a.distSq < b.distSq;
                                           });
            req = *bestIt;
            *bestIt = loadQueue_.back();
            loadQueue_.pop_back();
        }

        const std::filesystem::path filePath = sectorFilePath(sectorRoot_, req.coord);
        std::error_code ec;
        const bool exists = std::filesystem::exists(filePath, ec);
        std::optional<Result<SectorPackage>> loadResult;
        if (exists && !ec) {
            loadResult = loadSectorPackage(filePath);
        }

        if (stopToken.stop_requested()) {
            return;
        }

        std::scoped_lock lock(queueMutex_);
        const auto pit = pendingMap_.find(req.coord);
        if (pit == pendingMap_.end() || pit->second.generation != req.generation) {
            // Cancelled or superseded by a newer request while reading from disk.
            continue;
        }
        const HlodLevel latestDesiredLod = pit->second.targetLod;
        pendingMap_.erase(pit);

        if (!loadResult.has_value()) {
            absentSectors_.insert(req.coord);
            continue;
        }
        if (!loadResult->has_value()) {
            log::warn("WorldStreaming", "failed loading sector ({}, {}): {}", req.coord.x, req.coord.z,
                      loadResult->error().message);
            absentSectors_.insert(req.coord);
            continue;
        }

        if (latestDesiredLod == HlodLevel::Unloaded) {
            continue;
        }

        stagedQueue_.push_back(StagedSector{
            req.coord,
            latestDesiredLod,
            std::move(**loadResult),
        });
    }
}

} // namespace lumora
