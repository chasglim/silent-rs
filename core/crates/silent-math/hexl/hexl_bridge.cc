#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "hexl/hexl.hpp"

namespace chimera_hexl {
struct NttKey {
    std::uint64_t n;
    std::uint64_t modulus;
    std::uint64_t root;
};

struct NttKeyHash {
    std::size_t operator()(const NttKey& key) const noexcept {
        std::size_t h1 = std::hash<std::uint64_t>{}(key.n);
        std::size_t h2 = std::hash<std::uint64_t>{}(key.modulus);
        std::size_t h3 = std::hash<std::uint64_t>{}(key.root);
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct NttKeyEq {
    bool operator()(const NttKey& lhs, const NttKey& rhs) const noexcept {
        return lhs.n == rhs.n && lhs.modulus == rhs.modulus && lhs.root == rhs.root;
    }
};

static std::unordered_map<NttKey, intel::hexl::NTT, NttKeyHash, NttKeyEq> kCache;
static std::mutex kCacheMutex;

static intel::hexl::NTT& get_ntt(std::uint64_t n, std::uint64_t modulus, std::uint64_t root) {
    std::lock_guard<std::mutex> guard(kCacheMutex);
    NttKey key{n, modulus, root};
    auto it = kCache.find(key);
    if (it == kCache.end()) {
        it = kCache.emplace(key, intel::hexl::NTT(n, modulus, root)).first;
    }
    return it->second;
}
}  // namespace chimera_hexl

extern "C" {
void chimera_hexl_eltwise_add_mod(std::uint64_t* result, const std::uint64_t* op1,
                                 const std::uint64_t* op2, std::size_t n,
                                 std::uint64_t modulus) {
    intel::hexl::EltwiseAddMod(result, op1, op2, n, modulus);
}

void chimera_hexl_eltwise_sub_mod(std::uint64_t* result, const std::uint64_t* op1,
                                 const std::uint64_t* op2, std::size_t n,
                                 std::uint64_t modulus) {
    intel::hexl::EltwiseSubMod(result, op1, op2, n, modulus);
}

void chimera_hexl_eltwise_mul_mod(std::uint64_t* result, const std::uint64_t* op1,
                                 const std::uint64_t* op2, std::size_t n,
                                 std::uint64_t modulus, std::uint64_t input_mod_factor) {
    intel::hexl::EltwiseMultMod(result, op1, op2, n, modulus, input_mod_factor);
}

void chimera_hexl_ntt_forward(std::uint64_t* data, std::size_t n, std::uint64_t modulus,
                             std::uint64_t root, std::uint64_t input_mod_factor,
                             std::uint64_t output_mod_factor) {
    auto& ntt = chimera_hexl::get_ntt(n, modulus, root);
    ntt.ComputeForward(data, data, input_mod_factor, output_mod_factor);
}

void chimera_hexl_ntt_inverse(std::uint64_t* data, std::size_t n, std::uint64_t modulus,
                             std::uint64_t root, std::uint64_t input_mod_factor,
                             std::uint64_t output_mod_factor) {
    auto& ntt = chimera_hexl::get_ntt(n, modulus, root);
    ntt.ComputeInverse(data, data, input_mod_factor, output_mod_factor);
}
}
