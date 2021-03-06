// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with types that may require additional runtime checks for extensions
// supported by the device, e.g. 'half' or 'double'

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename SpecializationKernelName, typename T, int Dim,
          access::mode Mode, class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  if (Mode == access::mode::read_write)
    (OutBuf.template get_access<access::mode::write>())[0] = Identity;

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
#ifdef TEST_SYCL2020_REDUCTIONS
    auto Redu = sycl::reduction(OutBuf, CGH, Identity, BOp);
#else
    accessor<T, Dim, Mode, access::target::global_buffer> Out(OutBuf, CGH);
    auto Redu = ONEAPI::reduction(Out, Identity, BOp);
#endif

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  T MaxDiff = 3 * std::numeric_limits<T>::epsilon() *
              std::fabs(ComputedOut + CorrectOut);
  if (std::fabs(static_cast<T>(ComputedOut - CorrectOut)) > MaxDiff) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << ", MaxDiff = " << MaxDiff
              << "\n";
    assert(0 && "Wrong value.");
  }
}

template <typename T> int runTests(const string_class &ExtensionName) {
  device D = default_selector().select_device();
  if (!D.is_host() && !D.has_extension(ExtensionName)) {
    std::cout << "Test skipped\n";
    return 0;
  }

  constexpr access::mode ReadWriteMode = access::mode::read_write;
#ifdef TEST_SYCL2020_REDUCTIONS
  // TODO: property::reduction::initialize_to_identity is not supported yet.
  // Thus only read_write mode is tested now.
  constexpr access::mode DiscardWriteMode = access::mode::read_write;
#else
  constexpr access::mode DiscardWriteMode = access::mode::discard_write;
#endif

  // Check some less standards WG sizes and corner cases first.
  test<class A, T, 1, ReadWriteMode, std::multiplies<T>>(0, 4, 4);
  test<class B, T, 0, DiscardWriteMode, ONEAPI::plus<T>>(0, 4, 64);

  test<class C, T, 0, ReadWriteMode, ONEAPI::minimum<T>>(getMaximumFPValue<T>(),
                                                         7, 7);
  test<class D, T, 1, access::mode::discard_write, ONEAPI::maximum<T>>(
      getMinimumFPValue<T>(), 7, 7 * 5);

  test<class E, T, 1, ReadWriteMode, ONEAPI::plus<>>(1, 3, 3 * 5);
  test<class F, T, 1, DiscardWriteMode, ONEAPI::minimum<>>(
      getMaximumFPValue<T>(), 3, 3);
  test<class G, T, 0, DiscardWriteMode, ONEAPI::maximum<>>(
      getMinimumFPValue<T>(), 3, 3);

  std::cout << "Test passed\n";
  return 0;
}
