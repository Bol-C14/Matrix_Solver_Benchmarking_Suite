//------------------------------------------------------------------------------
// GB_AxB_saxbit_generic_firsti64.c: C=A*B, C bitmap/full, FIRSTI mult
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is bitmap only.
// multiply op is GxB_FIRSTI_INT64 or GxB_FIRSTI1_INT64

#define GB_AXB_SAXPY_GENERIC_METHOD GB_AxB_saxbit_generic_firsti64 
#define GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE  0
#define GB_GENERIC_OP_IS_POSITIONAL            1
#define GB_GENERIC_FLIPXY                      0
#define GB_GENERIC_OP_IS_INT64                 1
#define GB_GENERIC_OP_IS_FIRSTI                1
#define GB_GENERIC_OP_IS_FIRSTJ                0
#define GB_GENERIC_OP_IS_FIRST                 0
#define GB_GENERIC_OP_IS_SECOND                0

#include "mxm/factory/GB_AxB_saxpy_generic_method.c"

