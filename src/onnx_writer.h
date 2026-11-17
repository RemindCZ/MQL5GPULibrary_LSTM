#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <initializer_list>
namespace dnonnx {
// A writer for the subset of ONNX IR v7 / opset 11 used by forecast models.
struct Proto {
    std::vector<uint8_t> bytes;
    void var(uint64_t n) {
        while (n >= 128) { bytes.push_back(static_cast<uint8_t>((n & 127) | 128)); n >>= 7; }
        bytes.push_back(static_cast<uint8_t>(n));
    }
    void number(int field, uint64_t n) { var(field * 8); var(n); }
    void data(int field, const void* src, size_t length) {
        var(field * 8 + 2); var(length);
        const auto p = static_cast<const uint8_t*>(src);
        if (length) bytes.insert(bytes.end(), p, p + length);
    }
    void text(int field,const std::string& s) { data(field,s.data(),s.size()); }
    void message(int field,const Proto& p) { data(field,p.bytes.data(),p.bytes.size()); }
};
inline Proto value(const std::string& name,std::initializer_list<int> dims) {
    Proto shape, tensor, type, out;
    for (int d : dims) { Proto n; n.number(1,d); shape.message(1,n); }
    tensor.number(1,1); tensor.message(2,shape); type.message(1,tensor);
    out.text(1,name); out.message(2,type); return out;
}
inline void initializer(Proto& g,const std::string& name,std::initializer_list<int> dims,
                 int dtype,const void* bytes,size_t length) {
    Proto t;
    for (int d : dims) t.number(1,d);
    t.number(2,dtype); t.text(8,name); t.data(9,bytes,length); g.message(5,t);
}
inline void metadata(Proto& model,const std::string& key,const std::string& val) {
    Proto pair; pair.text(1,key); pair.text(2,val); model.message(14,pair);
}
}
