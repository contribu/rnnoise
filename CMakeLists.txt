cmake_minimum_required(VERSION 3.9)

project(rnnoise C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(deps/gflags)
add_subdirectory(deps/googletest)

include_directories(
  include
  src
)

add_compile_definitions(RNNOISE_FLOAT_WEIGHT)


file(GLOB RNNOISE_SOURCES src/*)

add_executable(rnnoise_demo ${RNNOISE_SOURCES} examples/rnnoise_demo.c)
target_compile_definitions(rnnoise_demo PRIVATE TRAINING=0)

add_executable(denoise_training ${RNNOISE_SOURCES})
target_compile_definitions(denoise_training PRIVATE TRAINING=1)
target_link_libraries(denoise_training gflags)

file(GLOB TEST_SOURCES test/*)
add_executable(test ${RNNOISE_SOURCES} ${TEST_SOURCES})
target_compile_definitions(test PRIVATE TRAINING=0)
target_link_libraries(test gtest gtest_main)
