cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(resnet_training)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
	set(_venv_python "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
	if(EXISTS "${_venv_python}")
		execute_process(
			COMMAND ${_venv_python} -c "import torch; print(torch.utils.cmake_prefix_path)"
			OUTPUT_VARIABLE _torch_cmake_prefix
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE _torch_prefix_status
			ERROR_QUIET
		)
		if(_torch_prefix_status EQUAL 0 AND NOT _torch_cmake_prefix STREQUAL "")
			list(APPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")
			message(STATUS "Using Torch CMake prefix from .venv: ${_torch_cmake_prefix}")
		endif()
	endif()
endif()

find_package(Torch REQUIRED)
find_package(MPI QUIET)

add_library(distdl INTERFACE)
target_include_directories(distdl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(distdl INTERFACE "${TORCH_LIBRARIES}")
if(MPI_CXX_FOUND)
	target_link_libraries(distdl INTERFACE MPI::MPI_CXX)
	target_compile_definitions(distdl INTERFACE DISTDL_HAS_MPI=1)
	message(STATUS "MPI found: enabling distdl distributed ops")
else()
	message(STATUS "MPI not found: distributed ops will be disabled")
endif()

add_executable(resnet_training resnet_example.cpp)

target_link_libraries(resnet_training PRIVATE distdl)
