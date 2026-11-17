#pragma once
#include "DNForecast.h"
#include <memory>
#include <string>
#include <vector>

namespace forecast {
struct Shape {
    int steps, markets, horizons;
    int prices() const { return (steps+1)*markets; }
    int features() const { return steps*markets*3; }
    int outputs() const { return horizons*DN3_FIELDS; }
};
bool valid_window(int steps, int markets);
bool encode(int steps, int markets, const double* closes, float* output);
class Model {
public:
    virtual ~Model() = default;
    virtual std::vector<double> predict(const std::vector<float>& input) = 0;
};
int capabilities();
std::unique_ptr<Model> load_model(Shape shape, int backend, const std::string& contract,
                                  const std::vector<uint8_t>& model);
void check_output(const std::vector<double>& result);
}
