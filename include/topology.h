#pragma once
#include "model_validation.h"
#include <stdexcept>
#include <vector>

namespace dn {
struct NodeSpec {
    int activation = 0; // dense: linear, tanh, ReLU, sigmoid
    int merge = 0;      // 0 concatenate, 1 add (equal widths)
    std::vector<int> parents; // -1 is the external sequence input
};
inline int merged_width(const NodeSpec& node, int input, const std::vector<int>& widths) {
    if (input <= 0 || node.parents.empty() || node.parents.size() > 64 ||
        node.merge < 0 || node.merge > 1) throw std::runtime_error("Invalid graph inputs or merge mode");
    std::int64_t total = 0;
    for (int parent : node.parents) {
        if (parent < -1 || parent >= static_cast<int>(widths.size()))
            throw std::runtime_error("Graph edges must reference input -1 or an earlier node");
        int width = parent == -1 ? input : widths[parent];
        if (node.merge == 0) total += width;
        else if (total == 0) total = width;
        else if (total != width) throw std::runtime_error("Add merge requires equal widths");
        if (total > max_elements) throw std::runtime_error("Merged width overflow");
    }
    return static_cast<int>(total);
}
inline bool all_reach_output(const std::vector<NodeSpec>& nodes, int output) {
    if (output < 0 || output >= static_cast<int>(nodes.size())) return false;
    std::vector<bool> used(nodes.size(), false);
    used[output] = true;
    for (int i = output; i >= 0; --i) if (used[i]) {
        for (int p : nodes[i].parents) {
            if (p < -1 || p >= i) return false;
            if (p >= 0) used[p] = true;
        }
    }
    for (bool u : used) if (!u) return false;
    return true;
}
} // namespace dn
