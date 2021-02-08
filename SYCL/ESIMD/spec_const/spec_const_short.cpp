//==--------------- spec_const_short.cpp  - DPC++ ESIMD on-device test ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// XFAIL: *
// RUN: %clangxx-esimd -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda

#define DEF_VAL -30572
#define REDEF_VAL 24794
#define STORE 2

typedef short spec_const_t;
typedef short container_t;

#include "Inputs/spec_const_common.hpp"
