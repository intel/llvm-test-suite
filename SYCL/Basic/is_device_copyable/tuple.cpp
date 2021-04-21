// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// TODO: enable the type when it gets support on Windows. Right now it is
// reported as not trivially copyable and thus cannot be passed to device.
// UNSUPPORTED: windows || level_zero

// This test performs basic check of std::tuple type mentioned
// in SYCL-2020 SPEC as one of implicitly device copyable types.
// It is device copyable if the template arguments of std::tuple class
// are device copyable.

#include <CL/sycl.hpp>

#include <iostream>
#include <tuple>

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
    std::tuple<int, float, double, CustomStruct> Tuple(7, 3.14, 123.456,
                                                       CustomStruct{2, 55.1});
    std::tuple<> EmptyTuple1;
    std::tuple<> EmptyTuple2;

    auto OutAcc = OutBuf.get_access<access::mode::discard_write>(CGH);
    CGH.single_task<class A>([=]() {
      int Score = 0;
      Score += (std::get<0>(Tuple) == 7) ? 1 : 0;
      Score += (std::get<1>(Tuple) == static_cast<float>(3.14)) ? 1 : 0;
      Score += (std::get<2>(Tuple) == static_cast<double>(123.456)) ? 1 : 0;
      Score += (std::get<3>(Tuple) == CustomStruct{2, 55.1}) ? 1 : 0;
      Score += EmptyTuple1 == EmptyTuple2;
      OutAcc[0] = Score;
    });
  });
  int Score = (OutBuf.template get_access<access::mode::read>())[0];
  if (Score != 5) {
    std::cerr << "Tuple basic test case failed." << std::endl;
    std::cerr << "Expected score = 5, computed = " << Score << std::endl;
    return 1;
  }
  return 0;
}

int testAcc(queue &Q) {
  buffer<int, 1> OutBuf1(range<1>{1});
  buffer<float, 1> OutBuf2(range<1>{1});
  buffer<double, 1> OutBuf3(range<1>{1});

  Q.submit([&](handler &CGH) {
    using AccType1 = accessor<int, 1, access::mode::discard_write,
                              access::target::global_buffer>;
    using AccType2 = accessor<float, 1, access::mode::discard_write,
                              access::target::global_buffer>;
    using AccType3 = accessor<double, 1, access::mode::discard_write,
                              access::target::global_buffer>;
    auto Tuple = std::make_tuple(AccType1(OutBuf1, CGH), AccType2(OutBuf2, CGH),
                                 AccType3(OutBuf3, CGH));

    CGH.single_task<class B>([=]() {
      std::get<0>(Tuple)[0] = 1001;
      std::get<1>(Tuple)[0] = 123.456;
      std::get<2>(Tuple)[0] = 789.001;
    });
  });
  int Res1 = (OutBuf1.get_access<access::mode::read>())[0];
  float Res2 = (OutBuf2.get_access<access::mode::read>())[0];
  double Res3 = (OutBuf3.get_access<access::mode::read>())[0];
  int Score = Res1 == 1001 && Res2 == static_cast<float>(123.456) &&
              Res3 == static_cast<double>(789.001);
  if (!Score) {
    std::cerr << "Tuple acc test case failed." << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  int NumErrors = 0;

  queue Q;
  NumErrors += testBasic(Q);
  NumErrors += testAcc(Q);

  if (NumErrors)
    std::cerr << NumErrors << " test cases failed" << std::endl;
  else
    std::cout << "Test PASSED." << std::endl;

  return 0;
}
