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
  /opt/intel/ipp/include
)

if(APPLE)
link_directories(
  /opt/intel/ipp/lib
)
set(IPP_LIBS
  /opt/intel/ipp/lib/libippi.dylib
  /opt/intel/ipp/lib/libipps.dylib
  /opt/intel/ipp/lib/libippvm.dylib
  /opt/intel/ipp/lib/libippcore.dylib
)
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rpath /opt/intel/ipp/lib")
else()
link_directories(
  /opt/intel/ipp/lib/intel64
)
set(IPP_LIBS
  libippi.a
  libipps.a
  libippcore.a
  libippvm.a
)
endif()

add_compile_definitions(RNNOISE_FLOAT_WEIGHT)
add_compile_definitions(RNNOISE_IPP)

file(GLOB RNNOISE_SOURCES src/*)

add_executable(rnnoise_demo ${RNNOISE_SOURCES} examples/rnnoise_demo.c)
target_compile_definitions(rnnoise_demo PRIVATE TRAINING=0)
target_link_libraries(rnnoise_demo ${IPP_LIBS})

add_executable(denoise_training ${RNNOISE_SOURCES})
target_compile_definitions(denoise_training PRIVATE TRAINING=1)
target_link_libraries(denoise_training gflags ${IPP_LIBS})

file(GLOB TEST_SOURCES test/*)
add_executable(test ${RNNOISE_SOURCES} ${TEST_SOURCES})
target_compile_definitions(test PRIVATE TRAINING=0)
target_link_libraries(test gtest gtest_main ${IPP_LIBS})
