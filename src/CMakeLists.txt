set_source_path(THIS_DIR_CXX_SOURCES
    kernelinfo.cu
)

add_subdirectory(gpumpi)

if(UNC_HAS_GPUCCL)
    add_subdirectory(gpuccl)
endif()

if(UNC_HAS_GPUSHMEM)
    add_subdirectory(gpushmem)
endif()

set(UNICONN_CXX_SOURCES "${CXX_SOURCES}" "${THIS_DIR_CXX_SOURCES}")
set(UNICONN_CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}")

add_library(Uniconn STATIC
    "${UNICONN_CXX_SOURCES}"
    "${UNICONN_CUDA_SOURCES}"
    "${UNICONN_HEADERS}")

add_library(UNICONN::Uniconn ALIAS Uniconn)

target_include_directories(Uniconn PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_include_directories(Uniconn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

set_target_properties(Uniconn
    PROPERTIES
    VERSION ${UNICONN_VERSION})
# set_target_properties(Uniconn PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_features(Uniconn PUBLIC cxx_std_17)

target_link_libraries(Uniconn PUBLIC MPI::MPI_CXX)
target_include_directories(Uniconn PUBLIC ${MPI_CXX_INCLUDE_DIRS})


if (CRAY_CC_BIN)
# In Cray environments, add links to GTL and GDRCopy libs for CUDA-aware support
target_link_libraries(Uniconn PUBLIC ${CRAY_MPI_GTL_LIBRARY})
# target_link_libraries(Uniconn PUBLIC ${GDRCOPY_LIBRARY})
endif()
if(UNC_HAS_ROCM)
    set(HIP_CLANG_ROOT "${UNC_ROCM_PATH}/llvm")

    # #############################
    # FIXME: This bit is to fix errors in the HIP CMake config
    # file. Hopefully this can be removed in a future version of
    # ROCm.
    file(GLOB HIP_CLANG_INCLUDE_SEARCH_PATHS
        "${HIP_CLANG_ROOT}/lib/clang/*/include")
    find_path(HIP_CLANG_INCLUDE_PATH stddef.h
        HINTS "${HIP_CLANG_INCLUDE_SEARCH_PATHS}"
        NO_DEFAULT_PATH)

    file(GLOB HIP_CLANGRT_LIB_SEARCH_PATHS
        "${HIP_CLANG_ROOT}/lib/clang/*/lib/*")
    find_library(ACTUAL_CLANGRT_BUILTINS clangrt-builtins
        NAMES
        clang_rt.builtins
        clang_rt.builtins-x86_64
        PATHS
        "${HIP_CLANGRT_LIB_SEARCH_PATHS}")

    get_target_property(_HIP_HOST_LIBS hip::host INTERFACE_LINK_LIBRARIES)
    get_target_property(_HIP_DEVICE_LIBS hip::device INTERFACE_LINK_LIBRARIES)

    string(REPLACE
        "CLANGRT_BUILTINS-NOTFOUND"
        "${ACTUAL_CLANGRT_BUILTINS}"
        _NEW_HIP_HOST_LIBS
        "${_HIP_HOST_LIBS}")
    string(REPLACE
        "CLANGRT_BUILTINS-NOTFOUND"
        "${ACTUAL_CLANGRT_BUILTINS}"
        _NEW_HIP_DEVICE_LIBS
        "${_HIP_DEVICE_LIBS}")

    set_property(TARGET hip::host
        PROPERTY INTERFACE_LINK_LIBRARIES ${_NEW_HIP_HOST_LIBS})
    set_property(TARGET hip::device
        PROPERTY INTERFACE_LINK_LIBRARIES ${_NEW_HIP_DEVICE_LIBS})

    # END FIXME BLOCK
    # #############################
    target_link_libraries(Uniconn PUBLIC
    
        hip::host
        hip::device
        $<$<BOOL:${UNC_HAS_GPUCCL}>:rccl>
        $<$<BOOL:${UNC_HAS_GPUSHMEM}>:rocshmem::rocshmem>
        ${Roctracer_LIBRARIES})

    # HIP language files are detected by the ".hip" extension. Since
    # we're upcycling ".cu" files for now, we just set their LANGUAGE to
    # be HIP.
    get_target_property(UNC_HIP_SOURCES Uniconn SOURCES)
    list(FILTER UNC_HIP_SOURCES INCLUDE REGEX ".cu$")
    set_source_files_properties(${UNC_HIP_SOURCES}
        PROPERTIES LANGUAGE HIP)


    target_compile_options(Uniconn PUBLIC $<$<COMPILE_LANGUAGE:HIP>:
        --gpu-max-threads-per-block=1024
        -fgpu-rdc
    >)
    target_link_options(Uniconn PUBLIC $<$<COMPILE_LANGUAGE:HIP>:
        --hip-link
        -fgpu-rdc
    >)

    set_target_properties(Uniconn
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE)
        
elseif(UNC_HAS_CUDA)
    target_link_libraries(Uniconn PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
        $<$<BOOL:${UNC_HAS_NVPROF}>:CUDA::nvToolsExt>
        $<$<BOOL:${UNC_HAS_GPUCCL}>:cuda::nccl>
        $<$<BOOL:${UNC_HAS_GPUSHMEM}>:nvshmem::nvshmem>)

        
    target_compile_features(Uniconn PUBLIC cuda_std_17)
    set_target_properties(Uniconn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(Uniconn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        --maxrregcount=32
        >)
endif(UNC_HAS_ROCM)


# Header installation.
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/uniconn
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
install(
    FILES ${PROJECT_SOURCE_DIR}/include/Uniconn.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS Uniconn
    EXPORT UniconnTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT UniconnTargets
    NAMESPACE UNICONN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/uniconn
)
