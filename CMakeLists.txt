####################################################
############ INSTALLING CORRECT CMAKE ##############
####################################################
# Installing correct cmake version is easy!
# 1) Find the respective version here;
# https://github.com/Kitware/CMake/releases,
# and 2) replace the [x.xx.x] in the following
# commands with the version number (remove the
# brackets). For example, if you are installing
# CMake 3.22.1, replace [x.xx.x] with 3.22.1:

# wget https://github.com/Kitware/CMake/releases/download/v[x.xx.x]/cmake-[x.xx.x]-linux-x86_64.sh
# chmod +x ./cmake-[x.xx.x]-linux-x86_64.sh
# ./cmake-[x.xx.x]-linux-x86_64.sh
# sudo mv cmake-[x.xx.x]-linux-x86_64 /opt/cmake
# sudo ln -s /opt/cmake/bin/* /usr/local/bin/
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# begin /* Update Essentials version */
set(ESSENTIALS_VERSION_MAJOR 2)
set(ESSENTIALS_VERSION_MINOR 1)
set(ESSENTIALS_VERSION_PATCH 0)
# end /* Update Essentials version */

set(ESSENTIALS_VERSION "${ESSENTIALS_VERSION_MAJOR}.${ESSENTIALS_VERSION_MINOR}.${ESSENTIALS_VERSION_PATCH}")

# Select "Release" as the default build type.
# This can be altered by setting -DCMAKE_BUILD_TYPE
# in the command-line interface to Release or Debug.
# No reason to set CMAKE_CONFIGURATION_TYPES if it's 
# not a multiconfig generator. Also no reason mess 
# with CMAKE_BUILD_TYPE if it's a multiconfig generator.
# https://stackoverflow.com/a/31548693/5729690
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
else()
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Defaulting to Release build type")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
    # set the valid options for cmake-gui drop-down list
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
endif()

####################################################
############# DEFINE PROJECT OPTIONS ###############
####################################################

option(ESSENTIALS_BUILD_EXAMPLES
  "If on, builds the example graph applications."
  ON)

option(ESSENTIALS_AMD_BACKEND
  "If on, builds essentials with __HIP_PLATFORM_AMD__ set."
  OFF)

option(ESSENTIALS_NVIDIA_BACKEND
  "If on, builds essentials with CUDA backend for NVIDIA GPUs."
  OFF)

option(ESSENTIALS_METAL_BACKEND
  "If on, builds essentials with Apple Metal backend for macOS GPUs."
  OFF)

option(ESSENTIALS_FETCH_ROCM_LIBS
  "If on, fetches rocPRIM/rocThrust/hipCUB via sparse-checkout instead of using system packages."
  OFF)

####################################################
############### BACKEND VALIDATION #################
####################################################

set(_backend_count 0)
if(${ESSENTIALS_AMD_BACKEND})
  math(EXPR _backend_count "${_backend_count} + 1")
endif()
if(${ESSENTIALS_NVIDIA_BACKEND})
  math(EXPR _backend_count "${_backend_count} + 1")
endif()
if(${ESSENTIALS_METAL_BACKEND})
  math(EXPR _backend_count "${_backend_count} + 1")
endif()

if(_backend_count EQUAL 0)
  message(FATAL_ERROR "No backend specified. Select exactly one:\n"
    "  -DESSENTIALS_NVIDIA_BACKEND=ON\n"
    "  -DESSENTIALS_AMD_BACKEND=ON\n"
    "  -DESSENTIALS_METAL_BACKEND=ON")
endif()
if(_backend_count GREATER 1)
  message(FATAL_ERROR "Multiple backends specified. Select exactly one:\n"
    "  -DESSENTIALS_NVIDIA_BACKEND=ON\n"
    "  -DESSENTIALS_AMD_BACKEND=ON\n"
    "  -DESSENTIALS_METAL_BACKEND=ON")
endif()

if(${ESSENTIALS_METAL_BACKEND} AND NOT APPLE)
  message(FATAL_ERROR "Metal backend requires macOS (Apple platform).")
endif()

####################################################
############### BUILD/TEST OPTIONS #################
####################################################

if(${ESSENTIALS_NVIDIA_BACKEND} OR ${ESSENTIALS_METAL_BACKEND})
  option(ESSENTIALS_BUILD_TESTS
    "If on, builds the unit tests."
    ON)
else()
  option(ESSENTIALS_BUILD_TESTS
    "If on, builds the unit tests."
    OFF)
endif()

if(${ESSENTIALS_NVIDIA_BACKEND})
  option(ESSENTIALS_BUILD_BENCHMARKS
    "If on, builds essentials with benchmarking support."
    OFF)
else()
  option(ESSENTIALS_BUILD_BENCHMARKS
    "If on, builds essentials with benchmarking support."
    OFF)
endif()

####################################################
############# PROJECT LANGUAGE SETUP ###############
####################################################

if(${ESSENTIALS_NVIDIA_BACKEND})
  message(STATUS "Essentials building with NVIDIA backend.")
  set(TARGET_LANGUAGE CUDA)
  project(essentials
    VERSION ${ESSENTIALS_VERSION}
    LANGUAGES CXX C ${TARGET_LANGUAGE}
    DESCRIPTION "Programmable CUDA/C++ GPU Graph Analytics"
    HOMEPAGE_URL "https://github.com/gunrock/gunrock"
  )
elseif(${ESSENTIALS_METAL_BACKEND})
  message(STATUS "Essentials building with Metal backend.")
  project(essentials
    VERSION ${ESSENTIALS_VERSION}
    LANGUAGES CXX C OBJCXX
    DESCRIPTION "Programmable GPU Graph Analytics"
    HOMEPAGE_URL "https://github.com/gunrock/gunrock"
  )
else()
  message(STATUS "Essentials building with AMD backend.")
  set(TARGET_LANGUAGE HIP)
  project(essentials
    VERSION ${ESSENTIALS_VERSION}
    LANGUAGES CXX C ${TARGET_LANGUAGE}
    DESCRIPTION "Programmable CUDA/C++ GPU Graph Analytics"
    HOMEPAGE_URL "https://github.com/gunrock/gunrock"
  )
endif()

####################################################
############### SET SM ARCHITECTURE ################
####################################################

## Note: This applies to NVBench as well.
## Can be used for applications by extracting the
## CUDA_ARCHITECTURES property from the project.
## see cmake's get_target_properties()
if(${ESSENTIALS_NVIDIA_BACKEND})
  if(NOT CMAKE_CUDA_ARCHITECTURES)
#     set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_ARCHITECTURES 90)
    message(STATUS "Using default GPU Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "GPU Architecture (-DCMAKE_CUDA_ARCHITECTURES): ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
elseif(${ESSENTIALS_AMD_BACKEND})
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
    message(STATUS "Using default GPU Architecture: ${CMAKE_HIP_ARCHITECTURES}")
  else()
    message(STATUS "GPU Architecture (-DCMAKE_HIP_ARCHITECTURES): ${CMAKE_HIP_ARCHITECTURES}")
  endif()
  set(AMDGPU_TARGETS ${CMAKE_HIP_ARCHITECTURES})
  set(DEFAULT_AMDGPU_TARGETS AMDGPU_TARGETS)
  message(STATUS "AMDGPU_TARGETS: ${AMDGPU_TARGETS}")
elseif(${ESSENTIALS_METAL_BACKEND})
  message(STATUS "Metal backend: architecture determined at runtime by Apple GPU.")
endif()

# begin /* Dependencies directory */
set(PROJECT_DEPS_DIR externals)
# end /* Dependencies directory */

# begin /* Include cmake modules */
if(${ESSENTIALS_AMD_BACKEND})
  include(${PROJECT_SOURCE_DIR}/cmake/FetchROCMHIP.cmake)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchROCmLibs.cmake)
elseif(${ESSENTIALS_NVIDIA_BACKEND})
  include(${PROJECT_SOURCE_DIR}/cmake/FetchCCCL.cmake)
elseif(${ESSENTIALS_METAL_BACKEND})
  include(${PROJECT_SOURCE_DIR}/cmake/FetchMetalCpp.cmake)
endif()

if(NOT ${ESSENTIALS_METAL_BACKEND})
  include(${PROJECT_SOURCE_DIR}/cmake/FetchCXXOpts.cmake)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchNlohmannJson.cmake)
endif()
include(${PROJECT_SOURCE_DIR}/cmake/FetchCMakeModules.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FetchThreads.cmake)
# end /* Include cmake modules */

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored")

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored")

## Export compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

############ ADD LIBRARY: ESSENTIALS (HEADER-ONLY) ############
add_library(essentials INTERFACE)

####################################################
############### SET TARGET PROPERTIES ##############
####################################################
if(${ESSENTIALS_NVIDIA_BACKEND})
  set_target_properties(essentials 
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} # Set required architecture.
  )
elseif(${ESSENTIALS_METAL_BACKEND})
  set_target_properties(essentials
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
  )
else()
  set_target_properties(essentials 
    PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      HIP_STANDARD 17
      HIP_STANDARD_REQUIRED ON
      HIP_EXTENSIONS OFF
  )
endif()

# Add GIT SHA-1 definition
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/externals/cmake_modules-src/")
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

####################################################
############ TARGET COMPILER DEFINITIONS ###########
####################################################
if(${ESSENTIALS_NVIDIA_BACKEND})
  target_compile_definitions(essentials
    INTERFACE 
      SM_TARGET=${CMAKE_CUDA_ARCHITECTURES}
      ESSENTIALS_BACKEND="NVIDIA"
      __HIP_PLATFORM_NVIDIA__
      HIP_BACKEND=1
  )
elseif(${ESSENTIALS_METAL_BACKEND})
  target_compile_definitions(essentials
    INTERFACE
      ESSENTIALS_BACKEND="METAL"
      ESSENTIALS_METAL_BACKEND=1
  )
else()
  # For AMD, keep SM_TARGET as architecture identifier (e.g., gfx942)
  # HIP may define gfx942 as a macro, which conflicts with rocPRIM's enum usage
  # We'll handle this by undefining the macro in compatibility headers if needed
  target_compile_definitions(essentials
    INTERFACE 
      SM_TARGET=${CMAKE_HIP_ARCHITECTURES}
      ESSENTIALS_BACKEND="AMD"
      __HIP_PLATFORM_AMD__
      HIP_BACKEND=0
  )
endif()

target_compile_definitions(essentials
  INTERFACE 
    ESSENTIALS_VERSION=${ESSENTIALS_VERSION}
    GIT_SHA1="${GIT_SHA1}"
)

####################################################
############ TARGET COMPILE FEATURES ###############
####################################################
# Turn C++ Standard 17 ON.
target_compile_features(essentials INTERFACE cxx_std_17)
# set(CMAKE_CXX_EXTENSIONS OFF)

set(ESSENTIALS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

####################################################
############ TARGET INCLUDE DIRECTORIES ############
####################################################
target_include_directories(essentials
  INTERFACE ${ESSENTIALS_INCLUDE_DIR}
  INTERFACE ${CMAKE_MODULES_INCLUDE_DIR}
)

if(${ESSENTIALS_METAL_BACKEND})
  target_include_directories(essentials
    INTERFACE ${METAL_CPP_INCLUDE_DIR}
  )
elseif(${ESSENTIALS_NVIDIA_BACKEND})
  target_include_directories(essentials
    INTERFACE ${ESSENTIALS_INCLUDE_DIR}/gunrock/compat
    INTERFACE ${CXXOPTS_INCLUDE_DIR}
    INTERFACE ${NHLOMANN_JSON_INCLUDE_DIR}
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    INTERFACE ${CCCL_INCLUDE_DIR}
  )
else()
  target_include_directories(essentials
    INTERFACE ${ESSENTIALS_INCLUDE_DIR}/gunrock/compat
    INTERFACE ${CXXOPTS_INCLUDE_DIR}
    INTERFACE ${NHLOMANN_JSON_INCLUDE_DIR}
    INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    INTERFACE ${THRUST_INCLUDE_DIR}
    INTERFACE ${HIPCUB_INCLUDE_DIR}
    INTERFACE ${CUB_INCLUDE_DIR}
    INTERFACE ${LIBCUDACXX_INCLUDE_DIR}
  )
  # Add rocPRIM include directory if using sparse-checkout fallback
  # rocThrust depends on rocPRIM, so we need its include path
  if(DEFINED ROCPRIM_INCLUDE_DIR AND ROCPRIM_INCLUDE_DIR)
    target_include_directories(essentials
      INTERFACE ${ROCPRIM_INCLUDE_DIR}
    )
    message(STATUS "Added rocPRIM include directory: ${ROCPRIM_INCLUDE_DIR}")
  endif()
endif()

####################################################
############ TARGET LINK LIBRARIES #################
####################################################

if(${ESSENTIALS_METAL_BACKEND})
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
  message(STATUS "Metal framework: ${METAL_FRAMEWORK}")
  message(STATUS "Foundation framework: ${FOUNDATION_FRAMEWORK}")
  message(STATUS "QuartzCore framework: ${QUARTZCORE_FRAMEWORK}")

  target_link_libraries(essentials
    INTERFACE ${METAL_FRAMEWORK}
    INTERFACE ${FOUNDATION_FRAMEWORK}
    INTERFACE ${QUARTZCORE_FRAMEWORK}
    INTERFACE ${CMAKE_THREAD_LIBS_INIT}
  )
elseif(${ESSENTIALS_NVIDIA_BACKEND})
  target_link_libraries(essentials
    INTERFACE curand
    INTERFACE cuda
    INTERFACE cusparse
    INTERFACE ${CMAKE_THREAD_LIBS_INIT}
    -lstdc++fs
  )
else()
  target_link_directories(essentials
    INTERFACE ${ROCM_PATH}
    INTERFACE ${ROCM_PATH}/lib
    INTERFACE ${HIP_PATH}
    INTERFACE ${HIP_PATH}/lib
  )

  target_link_libraries(essentials
    INTERFACE hip::device
    INTERFACE hiprand
    INTERFACE hipsparse
    INTERFACE ${CMAKE_THREAD_LIBS_INIT}
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-lstdc++fs>
  )
  
  if(TARGET roc::rocthrust)
    target_link_libraries(essentials INTERFACE roc::rocthrust)
  endif()
endif()

####################################################
################# TARGET SOURCES ###################
####################################################
if(NOT ${ESSENTIALS_METAL_BACKEND})
  target_sources(essentials
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/gunrock/util/gitsha1make.c"
  )
endif()

####################################################
############## SET CXX & CUDA FLAGS ################
####################################################

set(CXX_FLAGS 
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4
    /wd4244
  >
  $<$<CXX_COMPILER_ID:GNU>:
    -Wall 
    -Wno-unused-result
    -Wno-unused-local-typedefs
    -Wno-strict-aliasing
    -Wno-unused-function
    -Wno-format-security
    # -Wextra
    # -Werror
    # -vvv
  >
  $<$<CXX_COMPILER_ID:AppleClang>:
    -Wall
    -Wno-unused-result
    -Wno-deprecated-builtins
    -Wno-ignored-attributes
  >
  $<$<CXX_COMPILER_ID:Clang>:
    -Wno-unused-result
    -Wno-deprecated-builtins
    -Wno-ignored-attributes
    "-Wno-\#pragma-messages"
  >
)

if(NOT ${ESSENTIALS_METAL_BACKEND})
  set(CUDA_RELEASE_FLAGS
    --expt-extended-lambda 
    --expt-relaxed-constexpr
    --use_fast_math
    # Suppress common benign CUDA warnings
    --diag-suppress=177
    --diag-suppress=550
    --diag-suppress=20014
    $<$<CXX_COMPILER_ID:MSVC>:
      -Xcompiler=/bigobj
      -Xcompiler=/Zc:__cplusplus
      -Xcompiler=/wd4244
    >
  )

  set(CUDA_DEBUG_FLAGS
    --expt-extended-lambda 
    --expt-relaxed-constexpr
    --ptxas-options -v
    --debug # Host debug
    --device-debug # Device debug
    $<$<CXX_COMPILER_ID:MSVC>:
      -Xcompiler=/bigobj
      -Xcompiler=/Zc:__cplusplus
      -Xcompiler=/wd4244
    >
  )

  set(HIP_FLAGS
    -munsafe-fp-atomics
    -Ugfx942
    -Ugfx950
    -Ugfx90a
    -Ugfx908
    -Ugfx906
    -Ugfx1030
    -Ugfx1100
    -Wno-unused-result
    -Wno-deprecated-builtins
    -Wno-ignored-attributes
    "-Wno-\#pragma-messages"
  )

  # Add libcudacxx and HIP include paths for AMD backend
  if(${ESSENTIALS_AMD_BACKEND})
    if(DEFINED ROCM_PATH)
      if(EXISTS "${ROCM_PATH}/include/hip")
        list(APPEND HIP_FLAGS "-I${ROCM_PATH}/include/hip")
      endif()
      if(EXISTS "${ROCM_PATH}/hip/include")
        list(APPEND HIP_FLAGS "-I${ROCM_PATH}/hip/include")
      endif()
    endif()
  endif()

  set(HIP_RELEASE_FLAGS
    ${HIP_FLAGS}
    -O3
  )

  set(HIP_DEBUG_FLAGS
    ${HIP_FLAGS}
    -O0
    -ggdb
  )
endif()

####################################################
############ TARGET COMPILE OPTIONS ################
####################################################

if(${ESSENTIALS_METAL_BACKEND})
  target_compile_options(essentials INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
  )
elseif(${ESSENTIALS_NVIDIA_BACKEND})
  target_compile_options(essentials INTERFACE 
    $<$<AND:$<COMPILE_LANGUAGE:CXX>>:${CXX_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:${CUDA_DEBUG_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:${CUDA_RELEASE_FLAGS}>
  )
else()
  target_compile_options(essentials INTERFACE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:DEBUG>>:${HIP_DEBUG_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<CONFIG:RELEASE>>:${HIP_RELEASE_FLAGS}>
  )
endif()

####################################################
################# COLLECT METRICS ##################
####################################################
option(ESSENTIALS_COLLECT_METRICS
  "If on, builds Essentials with metrics collection enabled."
  OFF)

if(ESSENTIALS_COLLECT_METRICS)
  target_compile_definitions(essentials INTERFACE ESSENTIALS_COLLECT_METRICS=1)
  set(ESSENTIALS_BUILD_TESTS OFF)
  set(ESSENTIALS_BUILD_BENCHMARKS OFF)
else()
  target_compile_definitions(essentials INTERFACE ESSENTIALS_COLLECT_METRICS=0)
endif(ESSENTIALS_COLLECT_METRICS)

####################################################
############ BUILD EXAMPLE APPLICATIONS ############
####################################################

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_EXAMPLES AND NOT ${ESSENTIALS_METAL_BACKEND})
  add_subdirectory(examples)
endif()

####################################################
################ BUILD UNIT TESTS  #################
####################################################

# Subdirectories for examples, testing and documentation
if(ESSENTIALS_BUILD_TESTS)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchGoogleTest.cmake)
  enable_testing()
  add_subdirectory(unittests)
endif(ESSENTIALS_BUILD_TESTS)

####################################################
################ BUILD BENCHMARKS  #################
####################################################

if(ESSENTIALS_BUILD_BENCHMARKS)
  # ... see https://github.com/NVIDIA/nvbench/issues/66
  set(NVBench_ENABLE_NVML OFF)
  include(${PROJECT_SOURCE_DIR}/cmake/FetchNVBench.cmake)
  add_subdirectory(benchmarks)
endif(ESSENTIALS_BUILD_BENCHMARKS)
