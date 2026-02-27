cmake_minimum_required(VERSION 3.20)
project(MQL5GPULibrary_LSTM_UnitTests LANGUAGES CXX)

add_executable(kernel_api_tests kernel_api_tests.cpp)
target_compile_features(kernel_api_tests PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(kernel_api_tests PRIVATE /W4 /permissive-)
else()
  target_compile_options(kernel_api_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "Spusteni: kernel_api_tests <cesta_k_MQL5GPULibrary_LSTM.dll>")
