#ifndef TIMER_H
#define TIMER_H

#include <chrono>

using namespace std::chrono;

template <class unit = nanoseconds>
class Timer {

    using clock = high_resolution_clock;

public:
    Timer() : t_start(clock::now()) { }

    void reset() { t_start = clock::now(); }

    void start() { reset(); }

    double elapsed() 
    {
        return duration_cast<unit>(clock::now() - t_start).count();
    }

private:
    clock::time_point t_start;
};

class cycles {};

template <> class Timer<cycles> {
public:
    Timer() : t_start(get_cycle_count()) { }

    void reset() { t_start = get_cycle_count(); }

    void start() { reset(); }

    double elapsed() 
    {
        return static_cast<double>(get_cycle_count() - t_start);
    }

private:
    unsigned long long int t_start;

    inline __attribute__((always_inline))
    unsigned long long int get_cycle_count() {
        unsigned int hi, lo;
        asm volatile("cpuid\n\t" "rdtsc" : "=a"(lo), "=d"(hi));
        return ((unsigned long long)lo)|(((unsigned long long)hi) << 32);
    }
};

#endif
