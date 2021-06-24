// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %CPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %GPU_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %GPU_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
// RUN: %ACC_RUN_PLACEHOLDER %t.out &> %t.txt || true
// RUN: %ACC_RUN_PLACEHOLDER FileCheck %s --input-file %t.txt
//
// CHECK:      {{.*}}assert_in_one_kernel.cpp:22: void kernelFunc(int *, int): global id: [{{[0-3]}},0,0], local id: [0,0,0]
// CHECK-SAME: Assertion `Buf[wiID] != 0 && "from assert statement"` failed.
// CHECK-NOT:  The test ended.

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using namespace cl::sycl;
using namespace cl::sycl::access;

void kernelFunc(int *Buf, int wiID) {
  Buf[wiID] = 0;
  assert(Buf[wiID] != 0 && "from assert statement");
}

void assertTest() {
  std::array<int, 4> Vec = {1, 2, 3, 4};
  cl::sycl::range<1> numOfItems{Vec.size()};
  cl::sycl::buffer<int, 1> Buf(Vec.data(), numOfItems);

  queue Q;
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class TheKernel>(
        numOfItems, [=](item<1> Item) { kernelFunc(&Acc[0], Item[0]); });
  });
  Q.wait();
}

int main(int Argc, const char *Argv[]) {

  assertTest();

  std::cout << "The test ended." << std::endl;
  return 0;
}
