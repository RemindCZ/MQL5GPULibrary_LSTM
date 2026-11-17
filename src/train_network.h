#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace dnfit {
using Vec = std::vector<float>;
constexpr int hidden = 16;
struct Net {
    int steps, markets, horizons;
    std::vector<int64_t> taps;
    Vec center, scale, target_scale, w1, b1, w2, b2;
    Vec low, median, high, variance;
    int inputs() const { return static_cast<int>(center.size()); }
    void forward(const float* normalized, Vec& state, Vec& head) const;
    Vec predict(const Vec& head) const;
};
std::vector<uint8_t> export_onnx(const Net& net,const std::string& contract);
}
