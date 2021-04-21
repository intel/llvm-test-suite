#include <CL/sycl.hpp>

#include <iostream>
#include <string_view>

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test performs basic check of std::basic_string_view type mentioned
// in SYCL-2020 SPEC as one of implicitly device copyable types.
// std::basic_string_view is device copyable as a viewer/container only,
// i.e. the content must be allocated as USM memory in order to be
// accessible on device.

using namespace cl::sycl;

template <typename T> class Names;

template <typename Name, typename T>
int test(size_t NSyms, usm::alloc AllocType, queue &Q) {
  auto Dev = Q.get_device();

  if (AllocType == usm::alloc::shared &&
      !Dev.get_info<info::device::usm_shared_allocations>())
    return 0;
  if (AllocType == usm::alloc::host &&
      !Dev.get_info<info::device::usm_host_allocations>())
    return 0;
  if (AllocType == usm::alloc::device &&
      !Dev.get_info<info::device::usm_device_allocations>())
    return 0;

  T *StrPtr =
      (T *)malloc(sizeof(T) * 2 * NSyms, Dev, Q.get_context(), AllocType);
  if (StrPtr == nullptr)
    return 0;
  if (AllocType == usm::alloc::device) {
    Q.submit([&](handler &CGH) {
       CGH.single_task<Names<Name>>([=]() {
         for (int I = 0; I < NSyms; I++) {
           StrPtr[I * 2] = StrPtr[I * 2 + 1] = 'a' + I;
         }
       });
     }).wait();
  } else {
    for (int I = 0; I < NSyms; I++) {
      StrPtr[I * 2] = StrPtr[I * 2 + 1] = 'a' + I;
    }
  }

  buffer<T, 1> StrBuf(NSyms);
  std::basic_string_view View(StrPtr, 2 * NSyms);

  // Compute.
  Q.submit([&](handler &CGH) {
     auto StrAcc = StrBuf.template get_access<access::mode::discard_write>(CGH);
     CGH.parallel_for<Name>(range<1>{NSyms}, [=](id<1> Id) {
       size_t I = Id.get(0);
       StrAcc[I] = View[I * 2];
     });
   }).wait();

  // Check correctness.
  auto StrAcc = StrBuf.template get_access<access::mode::read>();
  for (int I = 0; I < NSyms; I++) {
    T ExpectedSym = 'a' + I;
    if (StrAcc[I] != ExpectedSym) {
      std::cerr << "Wrong result at index = " << I
                << ", expected = " << ExpectedSym
                << ", computed = " << StrAcc[I] << std::endl;
      return 1;
    }
  }

  free(StrPtr, Q.get_context());
  return 0;
}

template <typename Name, typename T> int testUSM(size_t NSyms, queue &Q) {
  int Errors = 0;
  Errors += test<Name, T>(NSyms, usm::alloc::shared, Q);
  Errors += test<Name, T>(NSyms, usm::alloc::host, Q);
  Errors += test<Name, T>(NSyms, usm::alloc::device, Q);
  return Errors;
}

int main() {
  int NumErrors = 0;
  queue Q;
  NumErrors += testUSM<class A, char>(26, Q);

  if (NumErrors)
    std::cerr << NumErrors << " test cases failed" << std::endl;
  else
    std::cout << "Test PASSED." << std::endl;

  return NumErrors;
}
