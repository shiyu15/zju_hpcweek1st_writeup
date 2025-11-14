cmake_minimum_required(VERSION 3.10)
project(mc_tick)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_SOURCE_DIR}/world ${CMAKE_SOURCE_DIR}/tick ${CMAKE_SOURCE_DIR}/util ${CMAKE_SOURCE_DIR}/player)

# 启用OpenMP支持
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, version: ${OpenMP_CXX_VERSION}")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-private-field") # -fsanitize=address
set(CMAKE_CXX_FLAGS_DEBUG "-g -Ofast -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -ffast-math -march=native -mtune=native -funroll-loops -fomit-frame-pointer -ftree-vectorize")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

FILE(GLOB WORLS_SRC
    "world/*.cpp"
)

FILE(GLOB TICK_SRC
    "tick/*.cpp"
)

FILE(GLOB UTIL_SRC
    "util/*.cpp"
)


FILE(GLOB TEST_SRC
    "test/*.cpp"
)

add_library(mc_tick_lib SHARED
    ${WORLS_SRC}
    ${TICK_SRC}
    ${UTIL_SRC}
)

# 链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(mc_tick_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

foreach(test_file ${TEST_SRC})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file} ${WORLS_SRC} ${TICK_SRC} ${UTIL_SRC})
endforeach()