// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test performs basic check of std::pair type mentioned
// in SYCL-2020 SPEC as one of implicitly device copyable types.
// It is device copyable if the template arguments of std::pair class
// are device copyable.

#include <CL/sycl.hpp>

#include <iostream>
#include <utility>

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
    std::pair<int, float> Pair1(7, 3.14);
    std::pair<double, CustomStruct> Pair2(123.456, CustomStruct{2, 55.1});

    auto OutAcc = OutBuf.get_access<access::mode::discard_write>(CGH);
    CGH.single_task<class A>([=]() {
      int Score = 0;
      Score += (Pair1.first == 7) ? 1 : 0;
      Score += (Pair1.second == static_cast<float>(3.14)) ? 1 : 0;
      Score += (Pair2.first == static_cast<double>(123.456)) ? 1 : 0;
      Score += (Pair2.second == CustomStruct{2, 55.1}) ? 1 : 0;
      OutAcc[0] = Score;
    });
  });
  int Score = (OutBuf.template get_access<access::mode::read>())[0];
  if (Score != 4) {
    std::cerr << "Tuple basic test case failed." << std::endl;
    std::cerr << "Expected score = 5, computed = " << Score << std::endl;
    return 1;
  }
  return 0;
}
#if 0
// TODO: support of accessors passed in std::pair does not work yet.
int testAcc(queue &Q) {
  buffer<int, 1> OutBuf1(range<1>{1});
  buffer<double, 1> OutBuf2(range<1>{1});

  Q.submit([&](handler &CGH) {
    using AccType1 = accessor<int, 1, access::mode::discard_write,
                              access::target::global_buffer>;
    using AccType2 = accessor<double, 1, access::mode::discard_write,
                              access::target::global_buffer>;
    std::pair<AccType1, AccType2> Pair(AccType1(OutBuf1, CGH),
                                       AccType2(OutBuf2, CGH));

    CGH.single_task<class B>([=]() {
      Pair.first[0] = 1001;
      Pair.second[0] = 123.456;
    });
  });
  int Res1 = (OutBuf1.get_access<access::mode::read>())[0];
  float Res2 = (OutBuf2.get_access<access::mode::read>())[0];
  int Score = Res1 == 1001 && Res2 == static_cast<float>(123.456);
  if (!Score) {
    std::cerr << "Tuple acc test case failed." << std::endl;
    return 1;
  }
  return 0;
}
#endif

int main() {
  int NumErrors = 0;

  queue Q;
  NumErrors += testBasic(Q);
  //  NumErrors += testAcc(Q);

  if (NumErrors)
    std::cerr << NumErrors << " test cases failed" << std::endl;
  else
    std::cout << "Test PASSED." << std::endl;

  return 0;
}
