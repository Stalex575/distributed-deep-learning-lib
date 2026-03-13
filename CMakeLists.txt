cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(resnet_training)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(resnet_training resnet_example.cpp)

target_link_libraries(resnet_training "${TORCH_LIBRARIES}")
