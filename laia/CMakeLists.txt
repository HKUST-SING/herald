if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# add_compile_definitions("DEBUG")

aux_source_directory(src LAIA_SRC_LIST)

find_package(pybind11 2.6.0 CONFIG)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(Boost REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found")
else()
    pybind11_add_module(laia_cache ${LAIA_SRC_LIST})
    target_compile_features(laia_cache PRIVATE cxx_std_17)
    target_include_directories(laia_cache PUBLIC include)

    # target_link_libraries(laia_scheduler PUBLIC ps)
    target_link_libraries(laia_cache PUBLIC ${Boost_LIBRARIES} pthread rt)

    # enable test
    enable_testing()
    add_subdirectory(test)
endif()