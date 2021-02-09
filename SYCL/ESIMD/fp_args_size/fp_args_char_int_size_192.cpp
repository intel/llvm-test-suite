//==------- fp_args_char_int_size_192.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx-esimd -Xclang -fsycl-allow-func-ptr -std=c++14 -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

constexpr unsigned VL = 16;
constexpr unsigned SIZE = 192;

typedef char a_data_t;
typedef int b_data_t;
typedef int c_data_t;

#include "Inputs/fp_args_size_common.hpp"
