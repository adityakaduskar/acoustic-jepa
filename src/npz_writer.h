// ---------------------------------------------------------------------------
//  npz_writer.h  (header-only)
//
//  Minimal, dependency-free writer for NumPy .npz files (a stored/uncompressed
//  ZIP of .npy members). Factored out of jepa_dump.cpp so both jepa_dump.cpp
//  (monostatic) and swarm_dump.cpp (cross-view) share one verified writer.
//
//  Supported dtypes: <f4 (float32), <i4 (int32), <f2 (float16, RN rounding).
//  np.load(f) yields the members as keys without the .npy suffix.
// ---------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace npz {

// ---- half-precision (IEEE 754 binary16), round-to-nearest -----------------
inline uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (exp <= 0) {                                  // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u; int shift = 14 - exp;
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1u) h += 1;       // round
        return (uint16_t)(sign | h);
    } else if (exp >= 31) {                          // overflow -> inf
        return (uint16_t)(sign | 0x7c00u);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    if ((man >> 12) & 1u) h += 1;                    // round
    return h;
}

// ---- CRC-32 (zip) ----------------------------------------------------------
inline uint32_t crc32_buf(const uint8_t* p, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1u)));
    }
    return ~crc;
}

// ---- one .npy member (header + raw little-endian data) --------------------
struct NpyArray { std::string name; std::vector<uint8_t> bytes; };

inline NpyArray make_npy(const std::string& name, const char* descr,
                         const std::vector<int>& shape, const uint8_t* data, size_t nbytes) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) { sh += std::to_string(shape[i]); if (i + 1 < shape.size()) sh += ", "; }
    if (shape.size() == 1) sh += ",";
    sh += ")";
    std::string hdr = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t base = 10 + hdr.size() + 1;                       // magic(6)+ver(2)+len(2)+hdr+'\n'
    hdr.append((64 - (base % 64)) % 64, ' '); hdr += "\n";
    uint16_t hlen = (uint16_t)hdr.size();

    NpyArray a; a.name = name;
    const uint8_t magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    a.bytes.insert(a.bytes.end(), magic, magic + 8);
    a.bytes.push_back(hlen & 0xff); a.bytes.push_back((hlen >> 8) & 0xff);
    a.bytes.insert(a.bytes.end(), hdr.begin(), hdr.end());
    a.bytes.insert(a.bytes.end(), data, data + nbytes);
    return a;
}

// ---- write a set of .npy members as a stored .npz --------------------------
inline void write_npz(const std::string& path, const std::vector<NpyArray>& arrays) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    struct CD { uint32_t crc, size, off; std::string name; };
    std::vector<CD> cds; uint32_t offset = 0;
    auto w16 = [&](uint16_t v) { std::fputc(v & 0xff, f); std::fputc((v >> 8) & 0xff, f); offset += 2; };
    auto w32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) std::fputc((v >> (8 * i)) & 0xff, f); offset += 4; };
    auto wr  = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); offset += (uint32_t)n; };

    for (const auto& a : arrays) {
        uint32_t crc = crc32_buf(a.bytes.data(), a.bytes.size());
        uint32_t loc = offset;
        w32(0x04034b50); w16(20); w16(0); w16(0);            // sig, ver, flags, method=stored
        w16(0); w16(0);                                      // time, date
        w32(crc); w32((uint32_t)a.bytes.size()); w32((uint32_t)a.bytes.size());
        w16((uint16_t)a.name.size()); w16(0);                // name len, extra len
        wr(a.name.data(), a.name.size());
        wr(a.bytes.data(), a.bytes.size());
        cds.push_back({ crc, (uint32_t)a.bytes.size(), loc, a.name });
    }
    uint32_t cdstart = offset;
    for (const auto& c : cds) {
        w32(0x02014b50); w16(20); w16(20); w16(0); w16(0);   // sig, vermade, verneed, flags, method
        w16(0); w16(0);                                      // time, date
        w32(c.crc); w32(c.size); w32(c.size);
        w16((uint16_t)c.name.size()); w16(0); w16(0);        // name, extra, comment len
        w16(0); w16(0); w32(0);                              // disk, internal attr, external attr
        w32(c.off);                                          // local header offset
        wr(c.name.data(), c.name.size());
    }
    uint32_t cdsize = offset - cdstart;
    w32(0x06054b50); w16(0); w16(0);
    w16((uint16_t)cds.size()); w16((uint16_t)cds.size());
    w32(cdsize); w32(cdstart); w16(0);
    std::fclose(f);
}

// helpers to turn typed vectors into .npy members
inline NpyArray npy_f32(const std::string& n, const std::vector<int>& shape, const std::vector<float>& v) {
    return make_npy(n, "<f4", shape, (const uint8_t*)v.data(), v.size() * sizeof(float));
}
inline NpyArray npy_i32(const std::string& n, const std::vector<int>& shape, const std::vector<int32_t>& v) {
    return make_npy(n, "<i4", shape, (const uint8_t*)v.data(), v.size() * sizeof(int32_t));
}
inline NpyArray npy_f16(const std::string& n, const std::vector<int>& shape, const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]);
    return make_npy(n, "<f2", shape, (const uint8_t*)h.data(), h.size() * sizeof(uint16_t));
}

} // namespace npz
