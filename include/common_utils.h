//
// Created by Zachary on 2026/4/22.
//

#ifndef LIVOX_COLOR_COMMON_UTILS_H
#define LIVOX_COLOR_COMMON_UTILS_H
#include <chrono>
#include <thread>
namespace common_utils {
    inline long currentTimeNanoseconds() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    inline long currentTimeMilliseconds() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    inline long currentTimeMicroseconds() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    inline void sleepMilliseconds(long ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
}
#endif //LIVOX_COLOR_COMMON_UTILS_H
