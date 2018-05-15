cmake_minimum_required(VERSION 2.8)
project(mesh_voxelization)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${Boost_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
add_executable(voxelize main.cpp)
target_link_libraries(voxelize ${Boost_LIBRARIES} ${HDF5_CXX_LIBRARIES})

add_executable(read_hdf5 examples/read_hdf5.cpp)
target_link_libraries(read_hdf5 ${Boost_LIBRARIES} ${HDF5_CXX_LIBRARIES})