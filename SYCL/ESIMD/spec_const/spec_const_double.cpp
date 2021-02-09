//==--------------- spec_const_double.cpp  - DPC++ ESIMD on-device test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// XFAIL: windows
// RUN: %clangxx-esimd -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda

#define DEF_VAL 9.1029384756e+11
#define REDEF_VAL -1.4432211654e-10
#define STORE 1

typedef double spec_const_t;
typedef double container_t;

#include "Inputs/spec_const_common.hpp"
