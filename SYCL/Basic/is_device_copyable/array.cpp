// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// This test performs basic check of std::array type mentioned
// in SYCL-2020 SPEC as one of implicitly device copyable types.
// That means that std::array can be passed between the host
// and device or between devices.

#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <numeric>

using namespace cl::sycl;

template <typename T, size_t N, size_t TestType = 0> class Names;

template <typename T, size_t N> int testBasic(queue &Q) {
  buffer<int, 1> OutBuf(range<1>{1});
  Q.submit([&](handler &CGH) {
    std::array<T, N> Arr;
    if constexpr (N > 0)
      std::iota(Arr.begin(), Arr.end(), 0);

    auto OutAcc = OutBuf.template get_access<access::mode::discard_write>(CGH);
    CGH.single_task<Names<T, N>>([=]() {
      int Score = 0;
      if constexpr (N > 0) {
        for (int I = 0; I < N; I++)
          Score += Arr[I] == I ? 1 : 0;
      }
      Score += Arr.size() == N ? 1 : 0;
      OutAcc[0] = Score;
    });
  });
  int Score = (OutBuf.template get_access<access::mode::read>())[0];
  if (Score != N + 1) {
    std::cerr << "ArrayN test case failed." << std::endl;
    return 1;
  }
  return 0;
}

template <typename T> int testAcc(queue &Q) {
  buffer<T, 1> OutBuf1(range<1>{1});
  buffer<T, 1> OutBuf2(range<1>{1});
  buffer<T, 1> OutBuf3(range<1>{1});

  Q.submit([&](handler &CGH) {
    using AccType = accessor<T, 1, access::mode::discard_write,
                             access::target::global_buffer>;
    std::array<AccType, 3> Arr = {AccType(OutBuf1, CGH), AccType(OutBuf2, CGH),
                                  AccType(OutBuf3, CGH)};

    CGH.single_task<Names<T, 3, 1>>([=]() {
      Arr[0][0] = 1001;
      Arr[1][0] = 1002;
      Arr[2][0] = 1003;
    });
  });
  int Res1 = (OutBuf1.template get_access<access::mode::read>())[0];
  int Res2 = (OutBuf2.template get_access<access::mode::read>())[0];
  int Res3 = (OutBuf3.template get_access<access::mode::read>())[0];
  int Score = Res1 == 1001 && Res2 == 1002 && Res3 == 1003;
  if (!Score) {
    std::cerr << "ArrayN test case failed." << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  int NumErrors = 0;

  queue Q;
  NumErrors += testBasic<int, 0>(Q);
  NumErrors += testBasic<int, 1>(Q);
  NumErrors += testBasic<float, 8>(Q);

  // TODO: enable this test case when compiler can pass std::arrays
  // more efficiently. Right now it passes each of std::array elements
  // as 4 separate parameters and thus quickly exceeds the maximum
  // possible number of kernel parameters.
  // NumErrors += testBasic<float, 1024 * 8>(Q);

  NumErrors += testAcc<float>(Q);

  if (NumErrors)
    std::cerr << NumErrors << " test cases failed" << std::endl;
  else
    std::cout << "Test PASSED." << std::endl;

  return 0;
}
