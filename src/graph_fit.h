#pragma once
#include "checkpoint.h"
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace graphfit {
using Floats=std::vector<float>;
struct Layer {
    int type,width;
    float dropout;
    dn::NodeSpec node;
};
struct Config {
    bool cuda=false;
    int batch=32;
    double rate=.001;
    std::vector<Layer> layers;
};
struct Head {
    Floats scale,low,median,high,variance,temperature;
    Floats forecast(const double* raw) const;
};
class Engine {
public:
    virtual ~Engine()=default;
    virtual void epoch()=0;
    virtual void predict(const double* x,int rows,double* y)=0;
    virtual dn::ModelState weights()=0;
    virtual void save_best()=0;
    virtual void restore_best()=0;
};
Config read(const std::filesystem::path& path,int features,int horizons);
std::string device();
std::unique_ptr<Engine> gpu(const Config&,int,int,int,const double*,const double*,int);
std::unique_ptr<Engine> cpu(const Config&,int,int,int,const double*,const double*,int);
std::vector<uint8_t> export_graph(const dn::ModelState&,const std::string&,const Head&,const Floats&,
                                  const Floats&,const Floats&);
std::vector<uint8_t> fit(const Config&,const std::string&,int,int,int,int,const std::vector<int64_t>&,
                         const std::vector<double>&,std::vector<double>&,std::vector<double>&);
}
