
set_source_path(THIS_DIR_HEADERS
    kernelinfo.hpp
    communicator.hpp
    coordinator.hpp
    memory.hpp
    environment.hpp

    # allgather.hpp
    # allgatherv.hpp
    # allreduce.hpp
    # alltoall.hpp
    # alltoallv.hpp
    # barrier.hpp
    # broadcast.hpp
    # commstate.hpp
    # gather.hpp
    # gatherv.hpp
    # get.hpp
    # peertopeer.hpp
    # onesidedsync.hpp
    # put.hpp
    # reduce.hpp
    # request.hpp
    # scatter.hpp
    # scatterv.hpp
    # twosidedp2p.hpp
)

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
