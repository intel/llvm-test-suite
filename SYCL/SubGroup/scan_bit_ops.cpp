// UNSUPPORTED: cpu
// #2252 Disable until all variants of built-ins are available in OpenCL CPU
// runtime for every supported ISA
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test verifies correct handling of exclusive_scan and inclusive_scan
// sub-group algorithm used with integer bitwise OR, XOR, AND operations.

#include "scan.hpp"

template <typename SpecializationKernelName, typename T>
void check_bit_ops(queue &Queue, size_t G = 256, size_t L = 4) {
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORF>, T>(
      Queue, T(L), ONEAPI::bit_or<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORT>, T>(
      Queue, T(0), ONEAPI::bit_or<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORF>, T>(
      Queue, T(L), ONEAPI::bit_xor<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORT>, T>(
      Queue, T(0), ONEAPI::bit_xor<T>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDF>, T>(
      Queue, T(L), ONEAPI::bit_and<T>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDT>, T>(
      Queue, ~T(0), ONEAPI::bit_and<T>(), true, G, L);

  // Transparent operator functors.
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORFV>, T>(
      Queue, T(L), ONEAPI::bit_or<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ORTV>, T>(
      Queue, T(0), ONEAPI::bit_or<>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORFV>, T>(
      Queue, T(L), ONEAPI::bit_xor<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_XORTV>, T>(
      Queue, T(0), ONEAPI::bit_xor<>(), true, G, L);

  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDFV>, T>(
      Queue, T(L), ONEAPI::bit_and<>(), false, G, L);
  check_op<sycl_subgr<SpecializationKernelName, class KernelName_ANDTV>, T>(
      Queue, ~T(0), ONEAPI::bit_and<>(), true, G, L);
}

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }
  check_bit_ops<class A, int>(Queue);
  check_bit_ops<class B, unsigned int>(Queue);
  check_bit_ops<class C, unsigned>(Queue);
  check_bit_ops<class D, long>(Queue);
  check_bit_ops<class E, unsigned long>(Queue);
  check_bit_ops<class F, long long>(Queue);
  check_bit_ops<class G, unsigned long long>(Queue);
  std::cout << "Test passed." << std::endl;
  return 0;
}
