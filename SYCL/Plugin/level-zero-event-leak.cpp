// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.out
//
// CHECK-NOT: ---> LEAK

#include <CL/sycl.hpp>
using namespace cl;
int main(int argc, char **argv) {
  sycl::queue Q;
  const unsigned n_chunk = 129;
  for (int i = 0; i < n_chunk; i++)
    Q.single_task([=]() {});
  Q.wait();
  return 0;
}
