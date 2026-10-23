#include "model_validation.h"
#include "checkpoint.h"
#include <sstream>
#include <cstdlib>
#include <iostream>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "Failed: " #expr << '\n'; return EXIT_FAILURE; } } while (0)

int main() {
    CHECK(dn::product(32, 64, 8));
    CHECK(!dn::product(-1, 64));
    CHECK(!dn::product(0, 64));
    CHECK(!dn::product(2147483647, 2));
    CHECK(!dn::product(2147483647, 2147483647, 2147483647));
    CHECK(dn::layer(0, 64, 0.2));
    CHECK(!dn::layer(-1, 64, 0));
    CHECK(!dn::layer(8, 0, 0));
    CHECK(!dn::layer(8, 64, 1));
    CHECK(!dn::layer(8, 64, 0.999999999999));
    CHECK(!dn::layer(8, 64, std::numeric_limits<double>::quiet_NaN()));
    CHECK(!dn::layer(2147483647, 2147483647, 0));
    CHECK(dn::training(100, 0, 0.001, 0));
    CHECK(!dn::training(0, 0, 0.001, 0));
    CHECK(!dn::training(100, -1, 0.001, 0));
    CHECK(!dn::training(100, 0, 1e-300, 0));
    CHECK(!dn::training(100, 0, 0.001, -1));
    CHECK(!dn::training(100, 0, std::numeric_limits<double>::infinity(), 0));
    CHECK(!dn::finite_float(1e100));
    const std::string valid = "SEQMODEL_V3 SEQ_LEN 2 LAYERS 1 2 1 1 0 8 4 "
        "0 0 0 0 0 0 0 0 0 0 0 0 OUTPUT 1 1 0.0000000001 0";
    std::istringstream stream(valid);
    auto model = dn::read_checkpoint(stream);
    CHECK(model.layers[0].type == 2 && model.layers[0].weights.size() == 8);
    CHECK(model.weights[0] > 0 && model.weights[0] < 1e-9f);
    std::istringstream v1("LSTM_V1 SEQ_LEN 1 LSTM_LAYERS 1 1 1 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 OUTPUT 1 1 1 0");
    CHECK(dn::read_checkpoint(v1).layers[0].type == 2);
    std::istringstream v2("SEQMODEL_V2 SEQ_LEN 1 LAYERS 1 0 1 1 0 0 0 0 OUTPUT 1 1 1 0");
    CHECK(dn::read_checkpoint(v2).layers[0].type == 0);
    const std::string invalid[] = {
        "", "SEQMODEL_V3 SEQ_LEN 0 LAYERS 1", "SEQMODEL_V3 SEQ_LEN 1 LAYERS -1",
        "SEQMODEL_V3 SEQ_LEN 1 LAYERS 999999999",
        "SEQMODEL_V3 SEQ_LEN 1 LAYERS 1 4 1 1 0",
        "SEQMODEL_V3 SEQ_LEN 1 LAYERS 1 2 1 1 1",
        "SEQMODEL_V3 SEQ_LEN 1 LAYERS 1 2 1 1 0 999999999 4",
        "SEQMODEL_V3 SEQ_LEN 1 LAYERS 1 2 1 1 0 8 4 nan",
        "SEQMODEL_V2 SEQ_LEN 1 LAYERS 1 1 1 1 0",
        valid + " junk", valid.substr(0, valid.size() - 2)
    };
    for (const auto& text : invalid) {
        bool rejected = false;
        try { std::istringstream bad(text); dn::read_checkpoint(bad); }
        catch (const std::runtime_error&) { rejected = true; }
        CHECK(rejected);
    }
    return EXIT_SUCCESS;
}
