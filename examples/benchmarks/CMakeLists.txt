

set(MPI_SOURCES
  src/bandwidth_mpi.cu
  src/latency_mpi.cu
)

set(CCL_SOURCES
  src/bandwidth_gpuccl.cu
  src/latency_gpuccl.cu
)
set(SHMEM_SOURCES
  src/bandwidth_nvshmem_device.cu
  src/latency_nvshmem_device.cu
)
set(UNC_SOURCES
  src/bandwidth_uniconn.cu
  src/latency_uniconn.cu
)
if(UNC_HAS_CUDA)
set_source_files_properties(${MPI_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${CCL_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${SHMEM_SOURCES} PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(${UNC_SOURCES} PROPERTIES LANGUAGE CUDA)
elseif(UNC_HAS_ROCM)
  set_source_files_properties(${MPI_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${CCL_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${SHMEM_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${UNC_SOURCES} PROPERTIES LANGUAGE HIP)
endif(UNC_HAS_CUDA)


macro(configure_target target_name)
  # set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  if(UNC_HAS_CUDA)
    target_compile_features(${target_name} PUBLIC cuda_std_17)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target_name} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        --maxrregcount=32
        >)
  elseif(UNC_HAS_ROCM)
    set_source_files_properties(${target_name}
        PROPERTIES LANGUAGE HIP)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
        --gpu-max-threads-per-block=1024
    >)

    set_target_properties(${target_name}
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE)
  endif(UNC_HAS_CUDA)

  target_link_libraries(${target_name} PRIVATE Uniconn)
  endmacro(configure_target)


foreach(src ${MPI_SOURCES})
  string(REPLACE ".cu" "" _tmp_exe_name "${src}")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  add_executable(${_exe_name} ${src})
  configure_target(${_exe_name})
endforeach()

if(UNC_HAS_GPUCCL)
  foreach(src ${CCL_SOURCES})
    string(REPLACE ".cu" "" _tmp_exe_name "${src}")
    get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
    add_executable(${_exe_name} ${src})
    configure_target(${_exe_name})
  endforeach()
endif(UNC_HAS_GPUCCL)

if(UNC_HAS_GPUSHMEM)
  foreach(src ${SHMEM_SOURCES})
    string(REPLACE ".cu" "" _tmp_exe_name "${src}")
    get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
    add_executable(${_exe_name} ${src})
    configure_target(${_exe_name})
  endforeach()
endif(UNC_HAS_GPUSHMEM)

foreach(src ${UNC_SOURCES})
  string(REPLACE ".cu" "" _tmp_exe_name "${src}")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  string(CONCAT _exe_name_mpi "${_exe_name}" "_mpi")
  add_executable(${_exe_name_mpi} ${src})
  target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_MPI=1)
  target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_GPUCCL=0)
  target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_GPUSHMEM=0)
  target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_HOST_ONLY=1)
  target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
  target_compile_definitions(${_exe_name_mpi} PRIVATE UNICONN_USE_FULL_DEVICE=0)
  configure_target(${_exe_name_mpi})

  if(UNC_HAS_GPUCCL)
    string(CONCAT _exe_name_gpuccl "${_exe_name}" "_gpuccl")
    add_executable(${_exe_name_gpuccl} ${src})
    target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_MPI=0)
    target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_GPUCCL=1)
    target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_GPUSHMEM=0)
    target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_HOST_ONLY=1)
    target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
    target_compile_definitions(${_exe_name_gpuccl} PRIVATE UNICONN_USE_FULL_DEVICE=0)
    configure_target(${_exe_name_gpuccl})
  endif(UNC_HAS_GPUCCL)

  if(UNC_HAS_GPUSHMEM)
    string(CONCAT _exe_name_nvshmem "${_exe_name}" "_nvshmem_device")
    add_executable(${_exe_name_nvshmem} ${src})
    target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_MPI=0)
    target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_GPUCCL=0)
    target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_GPUSHMEM=1)
    target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_HOST_ONLY=0)
    target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_LIMITED_DEVICE=0)
    target_compile_definitions(${_exe_name_nvshmem} PRIVATE UNICONN_USE_FULL_DEVICE=1)
    configure_target(${_exe_name_nvshmem})
  endif(UNC_HAS_GPUSHMEM)
  
endforeach()