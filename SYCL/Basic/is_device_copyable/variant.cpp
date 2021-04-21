// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test performs basic check of std::variant<> and std::variant<Ts...>
// types mentioned in SYCL-2020 SPEC as implicitly device copyable types.
// It is device copyable if all template arguments of std::variant class
// are device copyable.

#include <CL/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <variant>

using namespace cl::sycl;

struct CustomStruct {
  uint16_t A;
  uint16_t B;
  bool operator==(const CustomStruct &RHS) const {
    return A == RHS.A && B == RHS.B;
  }
};

int testBasic(queue &Q) {
  buffer<int, 1> OutBuf(range<1>{1});
  Q.submit([&](handler &CGH) {
    std::variant<uint64_t, double> Var1 =
        static_cast<uint64_t>(0x3FF0000000000000LL);
    std::variant<uint64_t, double> Var2 = static_cast<double>(3.14);
    std::variant<uint32_t, int32_t, float> Var3 = static_cast<float>(123.456);
    std::variant<std::monostate> VarEmpty{};

    auto OutAcc = OutBuf.template get_access<access::mode::discard_write>(CGH);
    CGH.single_task<class A>([=]() {
      // The code has to be trivial because std::get<>() method cannot be
      // used on DEVICE as it throws exception, not supported on DEVICE.
      OutAcc[0] = Var1.index() + Var2.index() * 10 + Var3.index() * 100 +
                  VarEmpty.index() * 1000;
      ;
    });
  });
  int ComputedRes = (OutBuf.template get_access<access::mode::read>())[0];
  int ExpectedRes = 210;
  if (ComputedRes != ExpectedRes) {
    std::cerr << "std::variant - basic test case failed. Expected = "
              << ExpectedRes << ", Computed = " << ComputedRes << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  int NumErrors = 0;

  queue Q;
  NumErrors += testBasic(Q);

  if (NumErrors)
    std::cerr << NumErrors << " test cases failed" << std::endl;
  else
    std::cout << "Test PASSED." << std::endl;

  return 0;
}
