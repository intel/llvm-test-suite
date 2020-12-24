// RUN: %clangxx %s -o %t1.out -lsycl -I %sycl_include
// RUN: %RUN_ON_HOST %t1.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: env SYCL_PI_TRACE=2 ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t2.out 2>&1 %GPU_CHECK_PLACEHOLDER

// RUN: %RUN_ON_HOST %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out

#include <CL/sycl.hpp>

#include <cassert>
#include <string>

using namespace cl::sycl;

int main() {
  const sycl::range<1> N{1};
  sycl::buffer<int, 1> Buf(
      N, {sycl::ext::oneapi::property::buffer::use_pinned_host_memory()});
  if (!Buf.has_property<
          sycl::ext::oneapi::property::buffer::use_pinned_host_memory>()) {
    std::cerr << "Buffer should have the use_pinned_host_memory property"
              << std::endl;
    return 1;
  }

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
    CGH.single_task<class init_a>([=]() {});
  });

  {
    int data1[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    {
      buffer<int, 1> a(data1, range<1>(10), {property::buffer::use_host_ptr()});
      buffer<int, 1> b(
          range<1>(10),
          {ext::oneapi::property::buffer::use_pinned_host_memory()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto A = a.get_access<access::mode::read_write>(cgh);
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class init_b>(range<1>{10}, [=](id<1> index) {
          B[index] = 0;
          A[index] = B[index] + 1;
        });
      });
    } // Data is copied back because there is a user side shared_ptr
    for (int i = 0; i < 10; i++)
      assert(data1[i] == 1);
  }

  try {
    int Data = 0;
    sycl::buffer<int, 1> Buf(
        &Data, N,
        {sycl::ext::oneapi::property::buffer::use_pinned_host_memory()});
    // Expected that exception is thrown
    return 1;
  } catch (sycl::invalid_object_error &E) {
    if (std::string(E.what()).find(
            "The use_pinned_host_memory cannot be used with host pointer") ==
        std::string::npos) {
      return 1;
    }

    return 0;
  }
}

// CHECK:---> piMemBufferCreate
// CHECK-NEXT: {{.*}} : {{.*}}
// CHECK-NEXT: {{.*}} : 17
// CHECK:ZE ---> zeMemAllocHost
