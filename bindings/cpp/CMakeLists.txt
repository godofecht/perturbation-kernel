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

# Name the static library explicitly rather than letting find_library
# pick. cargo emits both a staticlib and a cdylib, and on MSVC the
# cdylib's import library is `perturbation_kernel.dll.lib`, which
# find_library happily matches. Linking that produces an executable
# that needs the DLL at run time; with the DLL elsewhere, the process
# blocks on the loader instead of failing, which reads as a hung test.
if(WIN32)
  set(PK_RUST_LIB "${PK_RUST_TARGET_DIR}/perturbation_kernel.lib")
else()
  set(PK_RUST_LIB "${PK_RUST_TARGET_DIR}/libperturbation_kernel.a")
endif()

if(NOT EXISTS "${PK_RUST_LIB}")
  message(
    FATAL_ERROR
      "No static library at ${PK_RUST_LIB}.\n"
      "Run `cargo build --release` first, or point -DPK_RUST_TARGET_DIR at "
      "the directory holding it.")
endif()
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
# A hung test is worse than a failing one: it burns a CI runner to the
# job timeout and reports nothing. Fail it instead.
set_tests_properties(cpp_bindings PROPERTIES TIMEOUT 300)
