cmake_minimum_required(VERSION 3.24)
project(MQL5GPULibrary_LSTM VERSION 2.3.0 LANGUAGES CXX)

include(CTest)
option(DN_BUILD_CUDA "Build the Windows CUDA DLL" ON)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_executable(validation_tests tests/validation_tests.cpp)
    target_include_directories(validation_tests PRIVATE include)
    target_compile_features(validation_tests PRIVATE cxx_std_17)
    add_test(NAME validation COMMAND validation_tests)
    add_test(NAME api_contract COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_api.py")
    add_test(NAME financial_series COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_financial_series.py")
endif()

if(DN_BUILD_CUDA)
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "The MT5 DLL requires 64-bit Windows. Use DN_BUILD_CUDA=OFF for portable tests.")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(MQL5GPULibrary_LSTM SHARED kernel.cu)
    set_property(SOURCE kernel.cu PROPERTY OBJECT_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/model_validation.h;${CMAKE_CURRENT_SOURCE_DIR}/include/checkpoint.h;${CMAKE_CURRENT_SOURCE_DIR}/include/MQL5GPULibrary.h;${CMAKE_CURRENT_SOURCE_DIR}/include/topology.h;${CMAKE_CURRENT_SOURCE_DIR}/src/graph_layers.cuh")
    # NVIDIA's VS custom build does not reliably discover nested local headers.
    set_property(SOURCE kernel.cu PROPERTY VS_SETTINGS
        "AdditionalDependencies=${CMAKE_CURRENT_SOURCE_DIR}/include/model_validation.h$<SEMICOLON>${CMAKE_CURRENT_SOURCE_DIR}/include/checkpoint.h$<SEMICOLON>${CMAKE_CURRENT_SOURCE_DIR}/include/MQL5GPULibrary.h$<SEMICOLON>${CMAKE_CURRENT_SOURCE_DIR}/include/topology.h$<SEMICOLON>${CMAKE_CURRENT_SOURCE_DIR}/src/graph_layers.cuh")
    target_compile_features(MQL5GPULibrary_LSTM PRIVATE cxx_std_17 cuda_std_17)
    target_link_libraries(MQL5GPULibrary_LSTM PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
    set_target_properties(MQL5GPULibrary_LSTM PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    if(BUILD_TESTING)
        add_test(NAME gpu_api COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dll.py"
            "$<TARGET_FILE:MQL5GPULibrary_LSTM>" "${CUDAToolkit_BIN_DIR}")
        set_tests_properties(gpu_api PROPERTIES TIMEOUT 180 LABELS gpu)
    endif()
endif()
