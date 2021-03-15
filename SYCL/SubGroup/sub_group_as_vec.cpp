// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// Sub-groups are not suported on Host
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// Execution on CPU and FPGA takes 100000 times longer
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  cl::sycl::queue queue;
  printf("Device Name = %s\n",
         queue.get_device().get_info<cl::sycl::info::device::name>().c_str());

  // Initialize some host memory
  constexpr int N = 64;
  sycl::vec<int, 2> host_mem[N];
  for (int i = 0; i < N; ++i) {
    host_mem[i].s0() = i;
    host_mem[i].s1() = 0;
  }

  // Use the device to transform each value
  {
    cl::sycl::buffer<sycl::vec<int, 2>, 1> buf(host_mem, N);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto global =
          buf.get_access<cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>(cgh);
      sycl::accessor<sycl::vec<int, 2>, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local(N, cgh);
      cgh.parallel_for<class test>(
          cl::sycl::nd_range<1>(N, 32), [=](cl::sycl::nd_item<1> it) {
            cl::sycl::ONEAPI::sub_group sg = it.get_sub_group();
            if (!it.get_local_id(0)) {
              int end = it.get_global_id(0) + it.get_local_range()[0];
              for (int i = it.get_global_id(0); i < end; i++) {
                local[i].s0() = 0;
                local[i].s1() = i;
              }
            }
            it.barrier();

            int i = (it.get_global_id(0) / sg.get_max_local_range()[0]) *
                    sg.get_max_local_range()[0];
            // Global address space
            auto x = sg.load(&global[i]);

            // Local address space
            auto y = sg.load(&local[i]);

            sg.store(&global[i], x + y);
          });
    });
  }

  // Print results and tidy up
  for (int i = 0; i < N; ++i) {
    if (i != host_mem[i].s0() || i != host_mem[i].s1()) {
      printf("Unexpected result [%02d,%02d] vs [%02d,%02d]\n", i, i,
             host_mem[i].s0(), host_mem[i].s1());
      return 1;
    }
  }
  printf("Success!\n");
  return 0;
}
