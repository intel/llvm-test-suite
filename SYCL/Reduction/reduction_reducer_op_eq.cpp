// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test checks that if the custom type supports operations like +=, then
// such operations can be used for the reduction objects in kernels..

#include <CL/sycl.hpp>
#include <iostream>
#include <random>

using namespace sycl;
using namespace sycl::ONEAPI;

struct XY {
  XY() : X(0), Y(0) {}
  XY(float X, float Y) : X(X), Y(Y) {}
  float X;
  float Y;
  float x() const { return X; };
  float y() const { return Y; };
  XY &operator+=(const XY &RHS) {
    X += RHS.X;
    Y += RHS.Y;
    return *this;
  }
};

void setXY(float2 &Data, float X, float Y) { Data = {X, Y}; }
void setXY(XY &Data, float X, float Y) {
  Data.X = X;
  Data.Y = Y;
}

template <typename Name, typename T, typename BinaryOperation>
int test(T Identity, BinaryOperation BOp) {
  constexpr size_t N = 16;
  constexpr size_t L = 4;

  queue Q;
  T *Data = malloc_shared<T>(N, Q);
  T *Res = malloc_shared<T>(1, Q);
  T Expected = Identity;
  for (size_t I = 0; I < N; I++) {
    setXY(Data[I], I, I + 1);
    setXY(Expected, Expected.x() + I, Expected.y() + I + 1);
  }

  *Res = Identity;
  auto Red = reduction(Res, Identity, BOp);
  Q.submit([&](handler &H) {
     H.parallel_for<Name>(nd_range<1>{N, L}, Red,
                          [=](nd_item<1> ID, auto &Sum) {
                            size_t GID = ID.get_global_id(0);
                            Sum += Data[GID];
                          });
   }).wait();

  int Error = 0;
  if (Expected.x() != Res->x() || Expected.y() != Res->y()) {
    std::cerr << "Error: expected = (" << Expected.x() << ", " << Expected.y()
              << "); computed = (" << Res->x() << ", " << Res->y() << ")\n";
    Error = 1;
  }
  free(Res, Q);
  free(Data, Q);
  return Error;
}

int main() {
  int Error = 0;
  Error += test<class A, float2>(float2{}, std::plus<>{});
  Error += test<class B, XY>(
      XY{}, [](auto A, auto B) { return XY(A.X + B.X, A.Y + B.Y); });
  if (!Error)
    std::cout << "Passed.\n";
  return Error;
}
