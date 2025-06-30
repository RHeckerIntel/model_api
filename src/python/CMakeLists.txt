# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.26)

if(WIN32)
  set(CMAKE_GENERATOR_TOOLSET "v142")
endif()


add_subdirectory(../cpp model_api/cpp)

set(Python_FIND_VIRTUALENV FIRST)
project(_model_api LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


file(GLOB BINDINGS_SOURCES model_api/bindings/*.cpp)
file(GLOB BINDINGS_HEADERS model_api/bindings/*.hpp)

message(INFO ${BINDINGS_SOURCES})

nanobind_add_module(_model_api NB_STATIC STABLE_ABI LTO ${BINDINGS_SOURCES} ${BINDINGS_HEADERS})

set_target_properties(_model_api PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)
target_link_libraries(_model_api PRIVATE model_api)

install(TARGETS _model_api
    LIBRARY DESTINATION model_api  # Same place relative to package
)

install(
  DIRECTORY model_api
  DESTINATION .
  FILES_MATCHING PATTERN "*.py"
)
