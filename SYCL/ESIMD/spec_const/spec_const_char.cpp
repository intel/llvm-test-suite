//==--------------- spec_const_char.cpp  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows
// REQUIRES: linux && gpu
// RUN: %clangxx-esimd -fsycl -I%S/.. %s -o %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda

#define DEF_VAL -22
#define REDEF_VAL 33
#define STORE 2

typedef char spec_const_t;
typedef char container_t;

#include "Inputs/spec_const_common.hpp"
