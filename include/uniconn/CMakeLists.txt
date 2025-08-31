set_source_path(THIS_DIR_HEADERS
  gpumpi.hpp
  $<$<BOOL:${UNC_HAS_GPUCCL}>:gpuccl.hpp>
  $<$<BOOL:${UNC_HAS_GPUSHMEM}>:gpushmem.hpp>
  utils.hpp
)

add_subdirectory(interfaces)

add_subdirectory(gpumpi)

if(UNC_HAS_GPUCCL)
  add_subdirectory(gpuccl)
endif()

if(UNC_HAS_GPUSHMEM)
  add_subdirectory(gpushmem)
endif()

set(UNICONN_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)

# Note (trb 07/27/2020): The way generated files work in CMake is a
# bit tricky -- they only work properly if they're used in the same
# scope that they are created. So I'm moving all the install logic to
# "/src/CMakeLists.txt", including the header installation. This
# doesn't really matter for CUDA, but it's a problem for HIP/ROCm.
