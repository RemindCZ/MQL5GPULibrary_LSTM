#include "forecast_model.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

namespace {
using Id=int64_t;
using Clock=std::chrono::steady_clock;
struct Job {
    Id origin;
    std::vector<double> prices, result;
    int status=DN3_PENDING;
    bool running=false;
    std::string error;
};
struct Session {
    forecast::Shape shape;
    int backend, capacity, status=DN3_PENDING;
    bool latest, busy=false, closing=false;
    Id sequence=0;
    std::string contract,error;
    std::vector<uint8_t> bytes;
    std::unique_ptr<forecast::Model> model;
    std::map<Id,std::shared_ptr<Job>> jobs;
    std::array<Id,8> metrics{};
};
struct Runtime {
    std::mutex mutex,lifecycle;
    std::condition_variable changed;
    std::map<Id,std::shared_ptr<Session>> sessions;
    std::vector<std::thread> workers;
    Id sequence=0,cursor=0;
    bool stopping=false;

    void work() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            if (stopping) return;
            std::shared_ptr<Session> s;
            std::shared_ptr<Job> j;
            auto it=sessions.upper_bound(cursor);
            for (size_t n=0;n<sessions.size();++n,++it) {
                if (it==sessions.end()) it=sessions.begin();
                auto& candidate=it->second;
                if (candidate->busy || candidate->closing) continue;
                if (candidate->status==DN3_OK) {
                    for (auto& entry:candidate->jobs)
                        if (entry.second->status==DN3_PENDING) {j=entry.second;break;}
                    if (!j) continue;
                } else if (candidate->status!=DN3_PENDING) continue;
                cursor=it->first; s=candidate; break;
            }
            if (!s) {changed.wait(lock);continue;}
            s->busy=true;
            if (j) j->running=true;
            lock.unlock();
            const auto start=Clock::now();
            std::unique_ptr<forecast::Model> loaded;
            std::vector<double> result;
            std::string error;
            try {
                if (!j) loaded=forecast::load_model(s->shape,s->backend,s->contract,s->bytes);
                else {
                    std::vector<float> features(s->shape.features());
                    if (!forecast::encode(s->shape.steps,s->shape.markets,j->prices.data(),features.data()))
                        throw std::runtime_error("Invalid close price in input window");
                    result=s->model->predict(features);
                    forecast::check_output(result);
                }
            } catch (const std::exception& e) {error=e.what();}
              catch (...) {error="Unknown backend error";}
            lock.lock();
            if (!j) {
                s->error=std::move(error); s->model=std::move(loaded);
                s->status=s->model ? DN3_OK : DN3_FAILED;
                std::vector<uint8_t>().swap(s->bytes);
            } else {
                s->metrics[7]=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
                if (j->status==DN3_PENDING) {
                    j->result=std::move(result); j->error=std::move(error);
                    j->status=j->error.empty() ? DN3_OK : DN3_FAILED;
                    ++s->metrics[j->status==DN3_OK ? 1 : 5];
                }
                std::vector<double>().swap(j->prices);
                j->running=false;
            }
            s->busy=false;
            // Drop worker-owned references before waking the last closing client.
            s.reset(); j.reset();
            changed.notify_all();
        }
    }
    void stop_workers() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        changed.notify_all();
        for (auto& t:workers) if (t.joinable()) t.join();
        workers.clear();
        {std::lock_guard<std::mutex> lock(mutex);stopping=false;}
    }
    ~Runtime() { if (!workers.empty()) stop_workers(); }
};
Runtime& pool() {static Runtime p;return p;}

template<class F> auto safe(F f) noexcept -> decltype(f()) {
    try {return f();} catch (...) {return static_cast<decltype(f())>(DN3_FAILED);}
}
template<class F> int with_session(Id id,F f) {
    return safe([&]() -> int {
        auto& p=pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        auto it=p.sessions.find(id);
        if (it==p.sessions.end() || it->second->closing) return DN3_UNKNOWN;
        return f(*it->second);
    });
}
}

int DN3_GetAbiVersion() { return 30000; }
int DN3_GetCapabilities() { return safe([]{return forecast::capabilities();}); }
int DN3_Encode(int steps,int markets,const double* closes,int count,float* features,int length) {
    if (!forecast::valid_window(steps,markets) || !closes || !features || count!=(steps+1)*markets)
        return DN3_INVALID;
    if (length<steps*markets*3) return DN3_BUFFER;
    return forecast::encode(steps,markets,closes,features) ? DN3_OK : DN3_INVALID;
}
Id DN3_OpenSession(int steps,int markets,int horizons,int capacity,int latest,int backend,const char* contract,
                   int contract_size,const uint8_t* model,int model_size) {
    return safe([&]() -> Id {
        if (!forecast::valid_window(steps,markets) || horizons<1 || horizons>64 || horizons>steps/2 ||
            capacity<1 || capacity>256 || (latest!=0 && latest!=1) || backend<0 || backend>3 ||
            !contract || contract_size<1 || contract_size>2048 || std::memchr(contract,0,contract_size) ||
            model_size<0 || model_size>64*1024*1024 || ((backend==0)!=(model_size==0)) ||
            (model_size && !model)) return DN3_INVALID;
        auto s=std::make_shared<Session>();
        s->shape={steps,markets,horizons}; s->capacity=capacity; s->latest=latest!=0; s->backend=backend;
        s->contract.assign(contract,contract_size);
        if (model_size) s->bytes.assign(model,model+model_size);
        auto& p=pool();
        std::lock_guard<std::mutex> life(p.lifecycle);
        std::unique_lock<std::mutex> lock(p.mutex);
        if (p.sessions.size()>=64) return DN3_FULL;
        if (p.workers.empty()) {
            try {
                for (unsigned i=0;i<std::clamp(std::thread::hardware_concurrency(),1u,4u);++i)
                    p.workers.emplace_back([&p]{p.work();});
            } catch (...) {lock.unlock();p.stop_workers();throw;}
        }
        const Id id=++p.sequence;
        p.sessions.emplace(id,std::move(s));
        p.changed.notify_all();
        return id;
    });
}
int DN3_StatusOf(Id id) {return with_session(id,[](Session& s){return s.status;});}
int DN3_CloseSession(Id id) {
    return safe([&]() -> int {
        auto& p=pool();
        std::lock_guard<std::mutex> life(p.lifecycle);
        std::unique_lock<std::mutex> lock(p.mutex);
        auto it=p.sessions.find(id);
        if (it==p.sessions.end()) return DN3_UNKNOWN;
        auto s=it->second;
        s->closing=true;
        p.changed.wait(lock,[&]{return !s->busy;});
        p.sessions.erase(it);
        const bool last=p.sessions.empty();
        lock.unlock();
        if (last) p.stop_workers();
        return DN3_OK;
    });
}
Id DN3_Submit(Id id,Id origin,const double* closes,int count) {
    Id result=0;
    const int status=with_session(id,[&](Session& s) -> int {
        if (s.status!=DN3_OK) return s.status==DN3_PENDING ? DN3_INVALID : s.status;
        if (origin<=0 || !closes || count!=s.shape.prices()) return DN3_INVALID;
        if (s.jobs.size()>=static_cast<size_t>(s.capacity)) {++s.metrics[4];return DN3_FULL;}
        auto job=std::make_shared<Job>();
        job->origin=origin; job->prices.assign(closes,closes+count);
        result=++s.sequence;
        s.jobs.emplace(result,job);
        if (s.latest) for (auto& entry:s.jobs) {
            auto& old=*entry.second;
            if (entry.first!=result && !old.running && old.status==DN3_PENDING) {
                old.status=DN3_SUPERSEDED; ++s.metrics[2];
                std::vector<double>().swap(old.prices);
            }
        }
        ++s.metrics[0];pool().changed.notify_one();
        return DN3_OK;
    });
    return status==DN3_OK ? result : status;
}
int DN3_TryRead(Id id,Id job,double* out,int length,int* required,Id* origin) {
    return with_session(id,[&](Session& s) -> int {
        if (!required || !origin || length<0) return DN3_INVALID;
        auto it=s.jobs.find(job);
        if (it==s.jobs.end()) return DN3_UNKNOWN;
        auto& j=*it->second;
        *required=s.shape.outputs(); *origin=j.origin;
        if (j.status!=DN3_OK) return j.status;
        if (length<*required) return DN3_BUFFER;
        if (!out) return DN3_INVALID;
        std::copy(j.result.begin(),j.result.end(),out);
        return DN3_OK;
    });
}
int DN3_Cancel(Id id,Id job) {
    return with_session(id,[&](Session& s) -> int {
        auto it=s.jobs.find(job);
        if (it==s.jobs.end()) return DN3_UNKNOWN;
        auto& j=*it->second;
        if (j.status==DN3_PENDING) {
            j.status=DN3_CANCELLED; ++s.metrics[3];
            if (!j.running) std::vector<double>().swap(j.prices);
        }
        return j.status;
    });
}
int DN3_Release(Id id,Id job) {
    return with_session(id,[&](Session& s) -> int {
        if (job<=0 || job>s.sequence) return DN3_UNKNOWN;
        auto it=s.jobs.find(job);
        if (it!=s.jobs.end()) {
            if (it->second->status==DN3_PENDING) {it->second->status=DN3_CANCELLED;++s.metrics[3];}
            s.jobs.erase(it);
        }
        return DN3_OK;
    });
}
int DN3_Error(Id id,Id job,char* text,int length) {
    return with_session(id,[&](Session& s) -> int {
        const std::string* message=&s.error;
        if (job) {
            auto it=s.jobs.find(job);
            if (it==s.jobs.end()) return DN3_UNKNOWN;
            message=&it->second->error;
        }
        const int required=static_cast<int>(message->size()+1);
        if (length>=required && text) std::memcpy(text,message->c_str(),required);
        return required;
    });
}
int DN3_Metrics(Id id,Id* counters,int count) {
    return with_session(id,[&](Session& s) -> int {
        if (!counters || count<8) return DN3_BUFFER;
        s.metrics[6]=static_cast<Id>(s.jobs.size());
        std::copy(s.metrics.begin(),s.metrics.end(),counters);
        return DN3_OK;
    });
}
