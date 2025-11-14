cmake_minimum_required (VERSION 3.26)
project (ct_recon LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8.2-a -fopenmp")

find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable (ct_recon src/main.cpp src/fbp.cpp src/fbp.h)
target_include_directories(ct_recon PRIVATE ${HDF5_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ct_recon PRIVATE ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)