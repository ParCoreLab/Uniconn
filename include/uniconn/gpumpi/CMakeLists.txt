
set_source_path(THIS_DIR_HEADERS
    common.hpp
    communicator.hpp
    coordinator.hpp
    memory.hpp
    environment.hpp
)

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
