//==--------------- spec_const_ushort.cpp  - DPC++ ESIMD on-device test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// XFAIL: level_zero || windows
// RUN: %clangxx-esimd -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda

#define DEF_VAL 0xcafe
#define REDEF_VAL 0xdeaf
#define STORE 2

typedef unsigned short spec_const_t;
typedef unsigned short container_t;

#include "Inputs/spec_const_common.hpp"
