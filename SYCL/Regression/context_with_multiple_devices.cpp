// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=2 %GPU_RUN_PLACEHOLDER %t.out
// REQUIRES: level_zero && gpu

// Test the Level Zero backend with context having multiple root devices

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  auto Plt = platform(gpu_selector{});
  auto Devs = Plt.get_devices();
  auto Ctx = context(Devs);

  assert(Devs.size() == 2);

  queue Queue1{Ctx, Devs[0]};
  queue Queue2{Ctx, Devs[1]};

  int Arr[] = {2};
  {
    cl::sycl::buffer<int, 1> Buf(Arr, 1);
    Queue1.submit([&](cl::sycl::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel1>([=]() { Acc[0] *= 3; });
    });
    Queue2.submit([&](cl::sycl::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SimpleKernel2>([=]() { Acc[0] *= 3; });
    });
  }
  assert(Arr[0] == 18);

  return 0;
}
