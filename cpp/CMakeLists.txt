cmake_minimum_required(VERSION 3.22)
project(peik LANGUAGES C CXX CUDA)

# TODO EDF set these from command line instead
set(Python_ROOT_DIR "/home/eric/mambaforge/envs/pymir")
set(Python_INCLUDE_DIR "/home/eric/mambaforge/envs/pymir/include/python3.12")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
find_package(nanobind CONFIG REQUIRED)

include(FetchContent)


nanobind_add_module(eik1 eik1.cc)
target_compile_features(eik1 PUBLIC cxx_std_20)