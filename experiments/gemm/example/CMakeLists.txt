cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example gemm.cpp)
target_link_libraries(example ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})
set_property(TARGET example PROPERTY CXX_STANDARD 14)
