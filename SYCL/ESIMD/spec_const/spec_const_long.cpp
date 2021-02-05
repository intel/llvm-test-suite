//==--------------- spec_const_long.cpp  - DPC++ ESIMD on-device test -----===//
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

#define DEF_VAL -99776644220011
#define REDEF_VAL 22001144668855
#define STORE 1

typedef long spec_const_t;
typedef long container_t;

#include "Inputs/spec_const_common.hpp"
