// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %S/Inputs %s %S/Inputs/kernels_in_file2.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// CHECK:      {{.*}}kernels_in_file2.cpp:15: int calculus(int): global id: [5,0,0], local id: [1,0,0]
// CHECK-SAME: Assertion `X && "this message from calculus"` failed.
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.

#include "Inputs/kernels_in_file2.hpp"
#include <CL/sycl.hpp>
#include <iostream>

#ifdef DEFINE_NDEBUG_INFILE1
#define NDEBUG
#else
#undef NDEBUG
#endif

#include <cassert>

using namespace cl::sycl;
using namespace cl::sycl::access;

static constexpr size_t BUFFER_SIZE = 16;

int checkFunction() {
  int X = calculus(0);
  assert(X && "Nil in result");
  return X;
}

void enqueueKernel_1_fromFile1(queue *Q) {
  cl::sycl::range<1> numOfItems{BUFFER_SIZE};
  cl::sycl::buffer<int, 1> Buf(numOfItems);

  Q->submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class Kernel_1>(
        cl::sycl::nd_range(Buf.get_range(), cl::sycl::range<1>(4)),
        [=](cl::sycl::id<1> wiID) {
          int X = 0;
          if (wiID == 5)
            X = checkFunction();
          Acc[wiID] = X;
        });
  });
}

int main(int Argc, const char *Argv[]) {

  queue Q;
  enqueueKernel_1_fromFile1(&Q);
  enqueueKernel_2_fromFile2(&Q);
  Q.wait();

  std::cout << "The test ended." << std::endl;
  return 0;
}
