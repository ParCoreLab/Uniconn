cmake_minimum_required(VERSION 3.25)

# Version setup
set(UNICONN_VERSION_MAJOR 0)
set(UNICONN_VERSION_MINOR 1)
set(UNICONN_VERSION_PATCH 0)
set(UNICONN_VERSION
  "${UNICONN_VERSION_MAJOR}.${UNICONN_VERSION_MINOR}.${UNICONN_VERSION_PATCH}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Configuration type" FORCE)
endif()

project(UNICONN VERSION ${UNICONN_VERSION} LANGUAGES CXX)

if(${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are prohibited. "
    "Create a new directory and build there.")
endif()

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Options
option(UNICONN_ENABLE_CUDA "Enable CUDA support." OFF)
option(UNICONN_ENABLE_ROCM "Enable HIP/ROCm support." OFF)
option(UNICONN_ENABLE_GPUCCL "Enable GPUCCL support." OFF)
option(UNICONN_ENABLE_GPUSHMEM "Enable GPUSHMEM support." OFF)
option(UNICONN_ENABLE_DEBUG "Enable standard debugging sanity checks." OFF)
option(UNICONN_ENABLE_NVPROF "Enable profiling via nvprof/NVTX." OFF)
option(UNICONN_ENABLE_ROCTRACER "Enable profiling via rocprof/roctx." OFF)
option(UNICONN_ENABLE_TRACE "Enable runtime tracing." OFF)
option(UNICONN_ENABLE_TESTS "Build tests." OFF)
option(UNICONN_ENABLE_EXAMPLES "Build Examples." OFF)

# CUDA and ROCm are considered to be mutually exclusive.
if(UNICONN_ENABLE_CUDA AND UNICONN_ENABLE_ROCM)
  message(FATAL_ERROR
    "CUDA and HIP/ROCm support are mutually exclusive. "
    "Please only enable one.")
endif()

if(UNICONN_ENABLE_CUDA OR UNICONN_ENABLE_ROCM)
  set(UNICONN_HAS_GPU ON)
endif()

if(UNICONN_ENABLE_GPUCCL AND NOT UNICONN_HAS_GPU)
  message(FATAL_ERROR
    "GPUCCL support requested but no GPU runtime enabled.")
endif()

if(UNICONN_ENABLE_GPUSHMEM AND NOT UNICONN_HAS_GPU)
  message(FATAL_ERROR
    "GPUSHMEM support requested but no GPU runtime enabled. ")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" UNC_BUILD_TYPE_UPPER)

if(UNC_BUILD_TYPE_UPPER MATCHES "DEBUG")
  set(UNC_DEBUG ON)
endif()

if(UNICONN_ENABLE_DEBUG)
  set(UNC_DEBUG ON)
endif()

if(UNICONN_ENABLE_TRACE)
  set(UNC_TRACE ON)
endif()

if(UNICONN_ENABLE_TESTS)
  set(UNC_ENABLE_TESTS ON)
endif()

if(UNICONN_ENABLE_EXAMPLES)
  set(UNC_ENABLE_EXAMPLES ON)
endif()

# Setup CXX requirements
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# -faligned-new is needed to use new alignment-aware new when available.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-faligned-new" CXX_COMPILER_HAS_FALIGNED_NEW)

if(CXX_COMPILER_HAS_FALIGNED_NEW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()

# Set -g3 if supported to get better debugging experience.
check_cxx_compiler_flag("-g3" CXX_COMPILER_HAS_G3)

if(CXX_COMPILER_HAS_G3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

# Set -Og for debug builds when supported.
check_cxx_compiler_flag("-Og" CXX_COMPILER_HAS_OG)

if(CXX_COMPILER_HAS_OG)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -gdwarf-4")
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
add_definitions(-DOMPI_SKIP_MPICXX)


find_program(CRAY_CC_BIN "CC")

if (CRAY_CC_BIN)
  message(STATUS "Found Cray CC wrapper. Compiling for Cray programming environment.")
endif()



if(MPI_CXX_VERSION VERSION_GREATER_EQUAL 4.0)
  message(STATUS "MPI version >= 4.0, enabling large-count MPI support")
  set(UNC_HAS_LARGE_COUNT_MPI ON)
else()
  message(STATUS "MPI version < 4.0, disabling large-count MPI support")
  set(UNC_HAS_LARGE_COUNT_MPI OFF)
endif()


if(UNICONN_ENABLE_CUDA)

if (CRAY_CC_BIN)
  # FindMPI does not include Cray GTL (e.g. CUDA-aware) libs
  # automatically in Cray environment. Locate it to include in linking.
  string(REPLACE ":" ";" CRAY_LIB_PATHS $ENV{CRAY_LD_LIBRARY_PATH})
  find_library(CRAY_MPI_GTL_LIBRARY REQUIRED
    NAMES mpi_gtl_cuda
    HINTS ${CRAY_LIB_PATHS}
  )

  # Cray GTL libs benefit from linking against gdrcopy, so also
  # locating that library.
  # find_library(GDRCOPY_LIBRARY REQUIRED
  #   NAMES gdrapi
  # )

  message(STATUS "Found Cray GTL library: " ${CRAY_MPI_GTL_CUDA_LIBRARY})
  # message(STATUS "Found GDRCopy library: " ${GDRCOPY_LIBRARY})
endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
  endif()

  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)

  # If the previous find_package fails, we won't get here.
  set(UNC_HAS_CUDA TRUE)

  if(UNICONN_ENABLE_GPUCCL)
    find_package(NCCL 2.7.0 REQUIRED)
    set(UNC_HAS_GPUCCL TRUE)
  endif()

  if(UNICONN_ENABLE_GPUSHMEM)
    find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)
    set(UNC_HAS_GPUSHMEM TRUE)
  endif()

  if(UNICONN_ENABLE_NVPROF)
    if(NOT TARGET CUDA::nvToolsExt)
      find_package(NVTX REQUIRED)
      set(UNC_HAS_EXTERNAL_NVTX TRUE)
    endif()

    set(UNC_HAS_NVPROF ON)
  endif()
endif(UNICONN_ENABLE_CUDA)

if(UNICONN_ENABLE_ROCM)

if (CRAY_CC_BIN)
  # FindMPI does not include Cray GTL (e.g. CUDA-aware) libs
  # automatically in Cray environment. Locate it to include in linking.
  string(REPLACE ":" ";" CRAY_LIB_PATHS $ENV{CRAY_LD_LIBRARY_PATH})
  find_library(CRAY_MPI_GTL_LIBRARY REQUIRED
    NAMES mpi_gtl_hsa
    HINTS ${CRAY_LIB_PATHS}
  )

  # Cray GTL libs benefit from linking against gdrcopy, so also
  # locating that library.
  # find_library(GDRCOPY_LIBRARY REQUIRED
  #   NAMES gdrapi
  # )

  message(STATUS "Found Cray GTL library: " ${CRAY_MPI_GTL_LIBRARY})
  # message(STATUS "Found GDRCopy library: " ${GDRCOPY_LIBRARY})
endif()

  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx908 gfx90a)
  endif()
  # NOTE: This MUST come first. Even before the HIP language is
  # enabled. If it is not, then the header guard in
  # AMDDeviceLibsConfig.cmake will block future calls from finding the
  # correct AMD_DEVICE_LIBS_PREFIX, which will cause the command line
  # to have an invalid argument.
  find_package(hip REQUIRED)

  # Now that that's handled, we can enable HIP as a language.
  enable_language(HIP)

  list(APPEND UNC_HIP_HIPCC_FLAGS "-std=c++17 -fPIC")
  
  set(HIP_HIPCC_FLAGS "${UNC_HIP_HIPCC_FLAGS};${HIP_HIPCC_FLAGS}"
    CACHE STRING "Semi-colon delimited list of flags to pass to hipcc"
    FORCE) # <- because these are definitely the flags we want.

  if(ROCM_PATH)
    set(UNC_ROCM_PATH "${ROCM_PATH}")
  elseif(DEFINED ENV{ROCM_PATH})
    set(UNC_ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    # We could add a fallback here trying to find the base of some
    # likely-to-exist executable like "hipconfig" or "hipcc" or even
    # "rocm-smi" or something. Or the user can just use the ROCM_PATH
    # options above.
    set(UNC_ROCM_PATH "/opt/rocm")
  endif()

  message(STATUS "Using UNC_ROCM_PATH: ${UNC_ROCM_PATH}")

  # Provides hip_add_executable
  set(UNC_HAS_ROCM TRUE)

  if(UNICONN_ENABLE_GPUCCL)
    find_package(rccl 2.14.0 CONFIG REQUIRED QUIET
      HINTS ${RCCL_DIR} $ENV{RCCL_DIR} ${UNC_ROCM_PATH}
      PATH_SUFFIXES lib64/cmake/rccl lib/cmake/rccl
      NO_DEFAULT_PATH)

    message(STATUS "Found RCCL: ${rccl_DIR}")
    set(UNC_HAS_GPUCCL TRUE)
  endif()

  if(UNICONN_ENABLE_GPUSHMEM)
    find_package(rocshmem REQUIRED HINTS ${ROCSHMEM_PREFIX}
      PATH_SUFFIXES lib/cmake/rocshmem
      NO_DEFAULT_PATH)
    set(UNC_HAS_GPUSHMEM TRUE)
  endif()

  if(UNICONN_ENABLE_ROCTRACER)
    find_package(Roctracer MODULE COMPONENTS roctx)
    set(UNC_HAS_ROCTRACER ${Roctracer_FOUND})
  endif()
endif(UNICONN_ENABLE_ROCM)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
# Build library and executables

# Write the configuration file
# Update to include other profiling interfaces as needed:
if(UNC_HAS_NVPROF OR UNC_HAS_ROCTRACER)
  set(UNC_HAS_PROF ON)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/Unc_config.hpp.in"
  "${CMAKE_BINARY_DIR}/Unc_config.hpp" @ONLY)

# Macro for setting full paths to source files.
macro(set_source_path VAR)
  unset(__tmp_names)

  foreach(filename ${ARGN})
    list(APPEND __tmp_names "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach()

  set(${VAR} "${__tmp_names}")
endmacro()

add_subdirectory(include/uniconn)

# Add in the master header
list(APPEND UNICONN_HEADERS "${PROJECT_SOURCE_DIR}/include/Uniconn.hpp")

# Add the library targets
add_subdirectory(src)

#
# Install target
#
include(CMakePackageConfigHelpers)

# Build directory
export(EXPORT UniconnTargets NAMESPACE UNICONN:: FILE UniconnTargets.cmake)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/UniconnConfigVersion.cmake" VERSION
  ${UNICONN_VERSION} COMPATIBILITY SameMajorVersion)

set(INCLUDE_INSTALL_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
set(LIB_INSTALL_DIR src)
set(CMAKE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_LOCATION "${PROJECT_SOURCE_DIR}/cmake")
set(REAL_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(cmake/UniconnConfig.cmake.in
  "${CMAKE_BINARY_DIR}/UniconnConfig.cmake" INSTALL_DESTINATION
  "${CMAKE_INSTALL_DIR}" PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)
set(CMAKE_INSTALL_PREFIX "${REAL_CMAKE_INSTALL_PREFIX}")

# Install directory
set(INCLUDE_INSTALL_DIRS ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/uniconn)
set(CMAKE_MODULE_LOCATION "\$\{CMAKE_CURRENT_LIST_DIR\}")
configure_package_config_file(cmake/UniconnConfig.cmake.in
  "${CMAKE_BINARY_DIR}/UniconnConfig.cmake.install" INSTALL_DESTINATION
  ${CMAKE_INSTALL_DIR} PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Install the install-tree files
install(FILES "${CMAKE_BINARY_DIR}/UniconnConfig.cmake.install"
  RENAME "UniconnConfig.cmake" DESTINATION ${CMAKE_INSTALL_DIR})
install(FILES
  "${CMAKE_BINARY_DIR}/UniconnConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})
install(FILES
  "${CMAKE_BINARY_DIR}/Unc_config.hpp" DESTINATION ${INCLUDE_INSTALL_DIRS})

# Install the CMake modules we need
install(FILES
  cmake/FindNCCL.cmake
  cmake/FindRoctracer.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})

# ############################################
if(UNC_ENABLE_EXAMPLES)
  add_subdirectory(examples/benchmarks)
  add_subdirectory(examples/jacobi2D)
  if(UNC_HAS_CUDA)
    add_subdirectory(examples/CG-cuda)
  elseif(UNC_HAS_ROCM)
    add_subdirectory(examples/CG-hip)
  endif(UNC_HAS_CUDA)
endif(UNC_ENABLE_EXAMPLES)



