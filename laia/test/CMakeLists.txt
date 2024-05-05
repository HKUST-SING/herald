# aux_source_directory(./ LAIA_SRC_LIST)

find_package(pybind11 2.6.0 CONFIG)

if (NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found")
else()
    pybind11_add_module(laia_test test_array.cc ../src/utils.cc)
    target_compile_features(laia_test PRIVATE cxx_std_17)
    target_include_directories(laia_test PUBLIC ../include)
    
    add_executable(test_cpp_array test_cpp_array.cc ../src/array.cc)
    target_compile_features(test_cpp_array PRIVATE cxx_std_17)
    target_include_directories(test_cpp_array PUBLIC ../include)
    add_test(array_test test_cpp_array)
endif()