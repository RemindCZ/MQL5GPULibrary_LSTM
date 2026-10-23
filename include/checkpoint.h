#pragma once

#include "model_validation.h"
#include <istream>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace dn {
struct LayerState {
    // On-disk IDs: RNN=0, GRU=1, LSTM=2 (different from DN_AddLayerEx's selector).
    int type = 2, input = 0, hidden = 0;
    float dropout = 0;
    std::vector<float> weights, bias, candidate_x, candidate_h, bias_x, bias_h;
};
struct ModelState {
    int sequence = 0, output_input = 0, output = 0;
    std::vector<LayerState> layers;
    std::vector<float> weights, bias;
};

inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline void token(std::istream& input, const char* expected) {
    std::string value;
    require(static_cast<bool>(input >> value) && value == expected, "Invalid checkpoint field");
}
inline std::vector<float> values(std::istream& input, size_t count) {
    // Grow only as valid tokens arrive. A tiny malformed file cannot request a huge allocation.
    std::vector<float> result;
    for (size_t i = 0; i < count; ++i) {
        float value;
        require(static_cast<bool>(input >> value) && std::isfinite(value), "Missing or non-finite checkpoint weight");
        result.push_back(value);
    }
    return result;
}
inline ModelState read_checkpoint(std::istream& input) {
    input.imbue(std::locale::classic());
    ModelState state;
    std::string version;
    require(static_cast<bool>(input >> version), "Empty checkpoint");
    const bool v3 = version == "SEQMODEL_V3";
    const bool v1 = version == "LSTM_V1";
    require(v3 || v1 || version == "SEQMODEL_V2", "Unsupported checkpoint version");
    token(input, "SEQ_LEN");
    require(static_cast<bool>(input >> state.sequence) && state.sequence > 0, "Invalid sequence length");
    token(input, v1 ? "LSTM_LAYERS" : "LAYERS");
    int count;
    require(static_cast<bool>(input >> count) && count > 0 && count <= 1024, "Invalid layer count");
    for (int i = 0; i < count; ++i) {
        LayerState l;
        if (!v1) require(static_cast<bool>(input >> l.type), "Missing layer type");
        require(static_cast<bool>(input >> l.input >> l.hidden >> l.dropout), "Invalid layer header");
        require(l.type >= 0 && l.type <= 2 && l.input > 0 && layer(l.input, l.hidden, l.dropout), "Invalid layer dimensions or type");
        // V2 did not store the separated GRU candidate tensors. Never invent missing weights.
        require(v3 || l.type != 1, "Legacy GRU checkpoints lack candidate weights; export V3 instead");
        require(i == 0 || state.layers.back().hidden == l.input, "Disconnected recurrent layers");
        const size_t gates = static_cast<size_t>(l.hidden) * (l.type == 2 ? 4 : l.type == 1 ? 2 : 1);
        const size_t nw = static_cast<size_t>(l.input + l.hidden) * gates;
        if (v3) {
            size_t w, b;
            require(static_cast<bool>(input >> w >> b) && w == nw && b == gates, "Checkpoint tensor counts do not match architecture");
        }
        l.weights = values(input, nw);
        l.bias = values(input, gates);
        if (l.type == 1) {
            const size_t nx = static_cast<size_t>(l.input) * l.hidden;
            const size_t nh = static_cast<size_t>(l.hidden) * l.hidden;
            size_t a, b, c, d;
            require(static_cast<bool>(input >> a >> b >> c >> d) && a == nx && b == nh &&
                c == static_cast<size_t>(l.hidden) && d == c, "Invalid GRU candidate tensor counts");
            l.candidate_x = values(input, nx);
            l.candidate_h = values(input, nh);
            l.bias_x = values(input, c);
            l.bias_h = values(input, d);
        }
        state.layers.push_back(std::move(l));
    }
    token(input, "OUTPUT");
    require(static_cast<bool>(input >> state.output_input >> state.output) &&
        state.output_input == state.layers.back().hidden && product(state.output_input, state.output), "Invalid output dimensions");
    state.weights = values(input, static_cast<size_t>(state.output_input) * state.output);
    state.bias = values(input, state.output);
    input >> std::ws;
    require(input.eof(), "Unexpected trailing checkpoint data");
    return state;
}
} // namespace dn
