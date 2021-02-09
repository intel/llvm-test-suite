//==--------------- fp_call_from_func.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// XFAIL: *
// UNSUPPORTED: cuda
// RUN: %clangxx-esimd -Xclang -fsycl-allow-func-ptr -std=c++14 -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// The test checks that ESIMD kernels support use of function pointers from
// functions.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

#include <iostream>

class KernelID;

ESIMD_NOINLINE int add(int A, int B) { return A + B; }

template <typename AccTy> ESIMD_NOINLINE int test(AccTy acc, int A, int B) {
  using namespace sycl::INTEL::gpu;

  auto foo = &add;
  auto res = foo(A, B);

  scalar_store(acc, 0, res);
}

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  int result = 0;
  int *output = &result;

  int in1 = 100;
  int in2 = 233;

  {
    buffer<int, 1> buf(output, range<1>(1));

    auto e = q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<KernelID>(
          sycl::range<1>{1},
          [=](id<1> i) SYCL_ESIMD_KERNEL { test(acc, in1, in2); });
    });
    e.wait();
  }

  if (result != (in1 + in2)) {
    std::cout << "Failed" << std::endl;
    return 1;
  }

  return 0;
}
