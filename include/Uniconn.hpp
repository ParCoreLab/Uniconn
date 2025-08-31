//
// Created by dsagbili on 6/18/24.
//

#ifndef UNICONN_INCLUDE_UNICONN_H_
#define UNICONN_INCLUDE_UNICONN_H_

#include <Unc_config.hpp>
//#include "uniconn/utils.hpp"
#include "uniconn/gpumpi.hpp"

#if defined(UNC_HAS_GPUCCL)
#include "uniconn/gpuccl.hpp"
#endif

#if defined(UNC_HAS_GPUSHMEM)
#include "uniconn/gpushmem.hpp"
#endif

#endif  // UNICONN_INCLUDE_UNICONN_H_
