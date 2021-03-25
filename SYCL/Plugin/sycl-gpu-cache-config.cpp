// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t2.out

// UNSUPPORTED: opencl

#include <CL/sycl.hpp>

using namespace cl::sycl;


int main() {
  queue myQueue;
  property_list Props = {sycl::INTEL::property::kernel::gpu_cache_config(
      sycl::INTEL::gpu_cache_config::large_slm)};

  myQueue.submit(
      [&](handler &cgh) { cgh.single_task<class k8>([=]() {}, Props); });
  // CHECK: zeKernelSetCacheConfig

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k9>(
        range<1>(2), [=](id<1> index) {}, Props);
  });
  // CHECK: zeKernelSetCacheConfig

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k10>(
        range<2>(2, 2), [=](id<2> index) {}, Props);
  });
  // CHECK: zeKernelSetCacheConfig

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k11>(
        range<3>(2, 2, 2), [=](id<3> index) {}, Props);
  });
  // CHECK: zeKernelSetCacheConfig

  return 0;
}
