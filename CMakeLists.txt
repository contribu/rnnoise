cmake_minimum_required(VERSION 3.9)

project(rnnoise C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(deps/gflags)

include_directories(
  include
  src
)

file(GLOB RNNOISE_SOURCES src/*)
message("RNNOISE_SOURCES: ${RNNOISE_SOURCES}")

add_executable(rnnoise_demo ${RNNOISE_SOURCES} examples/rnnoise_demo.c)
target_compile_definitions(rnnoise_demo PRIVATE TRAINING=0)

add_executable(denoise_training ${RNNOISE_SOURCES})
target_compile_definitions(denoise_training PRIVATE TRAINING=1)
target_link_libraries(denoise_training gflags)
