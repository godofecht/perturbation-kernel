cmake_minimum_required(VERSION 3.16)
project(perturbation_kernel_cpp LANGUAGES CXX)

# C++20 for designated initialisers, which is what lets a family read as
# `Markov{.k = 5, .theta_max = 0.3}` rather than a bare argument list
# whose order you have to remember.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only interface target. Consumers link this and get the include
# path plus the static library built by cargo.
add_library(perturbation_kernel INTERFACE)
target_include_directories(perturbation_kernel
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Locate the staticlib cargo produced. PK_RUST_TARGET_DIR lets CI point
# at a non-default target directory.
set(PK_RUST_TARGET_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../target/release"
    CACHE PATH "Directory holding the compiled Rust static library")

find_library(
  PK_RUST_LIB
  NAMES perturbation_kernel libperturbation_kernel
  PATHS ${PK_RUST_TARGET_DIR}
  NO_DEFAULT_PATH REQUIRED)
message(STATUS "perturbation-kernel static library: ${PK_RUST_LIB}")

target_link_libraries(perturbation_kernel INTERFACE ${PK_RUST_LIB})

# The Rust staticlib needs the platform's system libraries. wgpu adds
# the graphics frameworks on Apple platforms when the gpu feature is on.
if(APPLE)
  target_link_libraries(
    perturbation_kernel
    INTERFACE "-framework CoreFoundation" "-framework Foundation"
              "-framework Metal" "-framework QuartzCore"
              "-framework CoreGraphics" "-framework IOKit" "-framework IOSurface"
              "-framework AppKit" objc iconv)
elseif(UNIX)
  target_link_libraries(perturbation_kernel INTERFACE pthread dl m)
elseif(WIN32)
  target_link_libraries(perturbation_kernel INTERFACE ntdll userenv ws2_32
                                                      bcrypt advapi32)
endif()

enable_testing()
add_executable(test_bindings tests/test_bindings.cpp)
target_link_libraries(test_bindings PRIVATE perturbation_kernel)
add_test(NAME cpp_bindings COMMAND test_bindings)
