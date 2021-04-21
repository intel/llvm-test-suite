// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test performs basic check of std::optional type mentioned
// in SYCL-2020 SPEC as one of implicitly device copyable types.

#include <CL/sycl.hpp>

#include <cstdint>
#include <iostream>
#include <optional>

using namespace cl::sycl;

struct CustomStruct {
  int A;
  double B;
  bool operator==(const CustomStruct &RHS) const {
    return A == RHS.A && B == RHS.B;
  }
};

int testBasic(queue &Q) {
  buffer<int, 1> OutBuf(range<1>{1});
  Q.submit([&](handler &CGH) {
    std::optional<uint64_t> OptInt(0xCAFEF00D);
    std::optional<uint64_t> OptEmpty{};
    std::optional<CustomStruct> OptCustom{CustomStruct{0x12345678, 123.456}};

    auto OutAcc = OutBuf.template get_access<access::mode::discard_write>(CGH);
    CGH.single_task<class A>([=]() {
      uint64_t GoodFood = OptInt.value_or(0xBAADF00D);
      uint64_t BaadFood = OptEmpty.value_or(0xBAADF00D);
      auto Custom = OptCustom.value_or(CustomStruct{0, 0.0});
      OutAcc[0] = GoodFood == 0xCAFEF00D && BaadFood == 0xBAADF00D &&
                  Custom == CustomStruct{0x12345678, 123.456};
    });
  });
  int Score = (OutBuf.template get_access<access::mode::read>())[0];
  if (!Score) {
    std::cerr << "optional basic test case failed." << std::endl;
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
