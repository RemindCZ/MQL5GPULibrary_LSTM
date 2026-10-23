#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace dn {
// Kernels and cuBLAS use signed 32-bit indices. Reserve headroom for the
// final grid-stride increment as well as launch rounding, not just allocations.
constexpr std::int64_t max_elements = std::numeric_limits<int>::max() / 2;

inline bool product(int a, int b, int c = 1) {
    return a > 0 && b > 0 && c > 0 &&
        static_cast<std::int64_t>(a) * b <= max_elements / c;
}

inline bool finite_float(double value) {
    return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max();
}

inline bool layer(int input, int hidden, double dropout) {
    return input >= 0 && hidden > 0 && finite_float(dropout) &&
        dropout >= 0.0 && dropout < 1.0 &&
        static_cast<float>(dropout) < 1.0f &&
        static_cast<std::int64_t>(input) + hidden <= max_elements &&
        hidden <= max_elements / 4 && product(input + hidden, hidden * 4);
}

inline bool training(int epochs, double target, double lr, double wd) {
    return epochs > 0 && epochs < std::numeric_limits<int>::max() &&
        finite_float(target) && target >= 0 && finite_float(lr) &&
        static_cast<float>(lr) > 0 && finite_float(wd) && wd >= 0;
}
} // namespace dn
