// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// TODO: Enable the test for HOST when it supports ONEAPI::reduce() and
// barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with USM var.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T1, typename T2> class KernelNameGroup;

template <bool IsSYCL2020Mode, typename T, typename BinaryOperation>
auto createReduction(T *USMPtr, T Identity, BinaryOperation BOp) {
  if constexpr (IsSYCL2020Mode)
    return sycl::reduction(USMPtr, Identity, BOp);
  else
    return ONEAPI::reduction(USMPtr, Identity, BOp);
}

template <typename Name, bool IsSYCL2020Mode, typename T, int Dim,
          class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems, usm::alloc AllocType) {
  queue Q;
  auto Dev = Q.get_device();

  if (AllocType == usm::alloc::shared &&
      !Dev.get_info<info::device::usm_shared_allocations>())
    return;
  if (AllocType == usm::alloc::host &&
      !Dev.get_info<info::device::usm_host_allocations>())
    return;
  if (AllocType == usm::alloc::device &&
      !Dev.get_info<info::device::usm_device_allocations>())
    return;

  T *ReduVarPtr = (T *)malloc(sizeof(T), Dev, Q.get_context(), AllocType);
  if (ReduVarPtr == nullptr)
    return;
  if (AllocType == usm::alloc::device) {
    event E = Q.submit([&](handler &CGH) {
      CGH.single_task<KernelNameGroup<Name, class Init>>(
          [=]() { *ReduVarPtr = Identity; });
    });
    E.wait();
  } else {
    *ReduVarPtr = Identity;
  }

  // Initialize.
  T CorrectOut;
  BinaryOperation BOp;

  buffer<T, 1> InBuf(NWItems);
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    auto Redu = createReduction<IsSYCL2020Mode>(ReduVarPtr, Identity, BOp);
    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<KernelNameGroup<Name, class Test>>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });
  Q.wait();

  // Check correctness.
  T ComputedOut;
  if (AllocType == usm::alloc::device) {
    buffer<T, 1> Buf(&ComputedOut, range<1>(1));
    event E = Q.submit([&](handler &CGH) {
      auto OutAcc = Buf.template get_access<access::mode::discard_write>(CGH);
      CGH.copy(ReduVarPtr, OutAcc);
    });
    ComputedOut = (Buf.template get_access<access::mode::read>())[0];
  } else {
    ComputedOut = *ReduVarPtr;
  }
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut
              << ", AllocMode: " << static_cast<int>(AllocType) << "\n";
    assert(0 && "Wrong value.");
  }

  free(ReduVarPtr, Q.get_context());
}

template <typename Name, typename T, int Dim, class BinaryOperation>
void testUSM(T Identity, size_t WGSize, size_t NWItems) {
  test<KernelNameGroup<Name, class SharedCase>, false, T, Dim, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::shared);
  test<KernelNameGroup<Name, class HostCase>, false, T, Dim, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::host);
  test<KernelNameGroup<Name, class DeviceCase>, false, T, Dim, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::device);

  test<KernelNameGroup<Name, class SharedCase2020>, true, T, Dim,
       BinaryOperation>(Identity, WGSize, NWItems, usm::alloc::shared);
  test<KernelNameGroup<Name, class HostCase2020>, true, T, Dim,
       BinaryOperation>(Identity, WGSize, NWItems, usm::alloc::host);
  test<KernelNameGroup<Name, class DeviceCase2020>, true, T, Dim,
       BinaryOperation>(Identity, WGSize, NWItems, usm::alloc::device);
}

int main() {
  // fast atomics and fast reduce
  testUSM<class AtomicReduce1, int, 1, ONEAPI::plus<int>>(0, 49, 49 * 5);
  testUSM<class AtomicReduce2, int, 0, ONEAPI::plus<int>>(0, 8, 128);

  // fast atomics
  testUSM<class Atomic1, int, 0, ONEAPI::bit_or<int>>(0, 7, 7 * 3);
  testUSM<class Atomic2, int, 1, ONEAPI::bit_or<int>>(0, 4, 128);

  // fast reduce
  testUSM<class Reduce1, float, 1, ONEAPI::minimum<float>>(
      getMaximumFPValue<float>(), 5, 5 * 7);
  testUSM<class Reduce2, float, 0, ONEAPI::maximum<float>>(
      getMinimumFPValue<float>(), 4, 128);

  // generic algorithm
  testUSM<class Generic1, int, 0, std::multiplies<int>>(1, 7, 7 * 5);
  testUSM<class Generic2, int, 1, std::multiplies<int>>(1, 8, 16);
  testUSM<class Generic3, CustomVec<short>, 0, CustomVecPlus<short>>(
      CustomVec<short>(0), 8, 8 * 3);

  std::cout << "Test passed\n";
  return 0;
}
