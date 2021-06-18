// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// RUN: %GPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// RUN: %ACC_RUN_PLACEHOLDER %t.out | FileCheck %s
//
// CHECK-NOT:  One shouldn't see this message
// CHECK:      {{.*}}assert_in_kernels.cpp:37: void kernelFunc2(int *, int): global id: [{{[0,2]}},0,0],
// CHECK-SAME: local id: [0,0,0] Assertion `Buf[wiID] == 0 && \"from assert statement\"` failed.
// CHECK-NOT:  test aborts earlier, one shouldn't see this message
// CHECK-NOT:  The test ended.

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using namespace cl::sycl;
using namespace cl::sycl::access;

void kernelFunc1(int *Buf, int wiID) {
  Buf[wiID] = 9;
  assert(Buf[wiID] != 0 && "One shouldn't see this message");
}

void assertTest1(queue &Q, buffer<int, 1> &Buf) {
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class Kernel_1>(
        Buf.get_range(),
        [=](cl::sycl::id<1> wiID) { kernelFunc1(&Acc[0], wiID); });
  });
}

void kernelFunc2(int *Buf, int wiID) {
  if (wiID % 2 != 0)
    Buf[wiID] = 0;
  assert(Buf[wiID] == 0 && "from assert statement");
}

void assertTest2(queue &Q, buffer<int, 1> &Buf) {
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class Kernel_2>(
        Buf.get_range(),
        [=](cl::sycl::id<1> wiID) { kernelFunc2(&Acc[0], wiID); });
  });
}

void kernelFunc3(int *Buf, int wiID) {
  if (wiID == 0)
    assert(false && "test aborts earlier, one shouldn't see this message");
  Buf[wiID] = 9;
}

void assertTest3(queue &Q, buffer<int, 1> &Buf) {
  Q.submit([&](handler &CGH) {
    auto Acc = Buf.template get_access<mode::read_write>(CGH);

    CGH.parallel_for<class Kernel_3>(
        Buf.get_range(),
        [=](cl::sycl::id<1> wiID) { kernelFunc3(&Acc[0], wiID); });
  });
}

int main(int Argc, const char *Argv[]) {
  std::array<int, 4> Vec = {1, 2, 3, 4};
  cl::sycl::range<1> numOfItems{Vec.size()};
  cl::sycl::buffer<int, 1> Buf(Vec.data(), numOfItems);

  queue Q;
  assertTest1(Q, Buf);
  Q.wait();

  assertTest2(Q, Buf);
  Q.wait();

  assertTest3(Q, Buf);
  Q.wait();

  std::cout << "The test ended." << std::endl;
  return 0;
}
