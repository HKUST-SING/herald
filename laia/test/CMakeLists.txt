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

    add_executable(test_ringbuffer test_ringbuffer.cc)
    target_compile_features(test_ringbuffer PRIVATE cxx_std_17)
    target_include_directories(test_ringbuffer PUBLIC ../include)
    target_link_libraries(test_ringbuffer ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread rt)

    add_executable(run_tests test_share_mem.cc) 
    target_include_directories(run_tests PUBLIC ../include)
    target_compile_features(run_tests PRIVATE cxx_std_17)
    target_link_libraries(run_tests ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread rt)
    target_link_libraries(run_tests ${Boost_LIBRARIES})
endif()