// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out

#include <CL/sycl.hpp>

using namespace cl::sycl;

class k0;
class k1;
class k2;
class k3;
class k4;
class k5;
class k6;
class k7;

int main() {
  queue myQueue;
  property_list Props = {};

  const range<1> globalRange(2);
  const range<1> localRange(2);
  const id<1> globalOffset(0);
  const nd_range<1> ndRange(globalRange, localRange, globalOffset);

  cl::sycl::program Prg0(myQueue.get_context());
  Prg0.build_with_kernel_type<k0>();
  cl::sycl::kernel Krn0 = Prg0.get_kernel<k0>();

  assert(Prg0.has_kernel<k0>());

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.single_task<k0>(
        Krn0, [=]() {}, Props);
  });

  myQueue.submit([&](cl::sycl::handler &Cgh) { Cgh.single_task(Krn0, Props); });

  cl::sycl::program Prg1(myQueue.get_context());
  Prg1.build_with_kernel_type<k1>();
  cl::sycl::kernel Krn1 = Prg1.get_kernel<k1>();

  assert(Prg1.has_kernel<k1>());

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.parallel_for<k1>(
        Krn1, range<1>(2), [=](id<1> index) {}, Props);
  });

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.parallel_for(range<1>(2), Krn1, Props);
  });

  cl::sycl::program Prg2(myQueue.get_context());
  Prg2.build_with_kernel_type<k2>();
  cl::sycl::kernel Krn2 = Prg2.get_kernel<k2>();

  assert(Prg2.has_kernel<k2>());

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.parallel_for<k2>(
        Krn2, range<2>(2, 2), [=](id<2> index) {}, Props);
  });

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.parallel_for(range<2>(2, 2), Krn2, Props);
  });

  cl::sycl::program Prg3(myQueue.get_context());
  Prg3.build_with_kernel_type<k3>();
  cl::sycl::kernel Krn3 = Prg3.get_kernel<k3>();

  assert(Prg3.has_kernel<k3>());

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.parallel_for<k3>(
        Krn3, range<3>(2, 2, 2), [=](id<3> index) {}, Props);
  });

  myQueue.submit([&](cl::sycl::handler &Cgh) {
    Cgh.parallel_for(range<3>(2, 2, 2), Krn3, Props);
  });
  cl::sycl::program Prg4(myQueue.get_context());
  Prg4.build_with_kernel_type<k4>();
  cl::sycl::kernel Krn4 = Prg4.get_kernel<k4>();

  assert(Prg4.has_kernel<k4>());

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<k4>(
        Krn4, range<1>(2), id<1>(0), [=](item<1> index) {}, Props);
  });

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for(range<1>(2), id<1>(0), Krn4, Props);
  });

  cl::sycl::program Prg5(myQueue.get_context());
  Prg5.build_with_kernel_type<k5>();
  cl::sycl::kernel Krn5 = Prg5.get_kernel<k5>();

  assert(Prg5.has_kernel<k5>());

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<k5>(
        Krn5, ndRange, [=](nd_item<1> index) {}, Props);
  });

  myQueue.submit([&](handler &cgh) { cgh.parallel_for(ndRange, Krn5, Props); });

  cl::sycl::program Prg6(myQueue.get_context());
  Prg6.build_with_kernel_type<k6>();
  cl::sycl::kernel Krn6 = Prg6.get_kernel<k6>();

  assert(Prg6.has_kernel<k6>());

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for_work_group<k6>(
        Krn6, range<1>(2), [=](group<1> g) {}, Props);
  });

  cl::sycl::program Prg7(myQueue.get_context());
  Prg7.build_with_kernel_type<k7>();
  cl::sycl::kernel Krn7 = Prg7.get_kernel<k7>();

  assert(Prg7.has_kernel<k7>());

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for_work_group<k7>(
        Krn7, range<1>(2), range<1>(2), [=](group<1> g) {}, Props);
  });

  myQueue.submit(
      [&](handler &cgh) { cgh.single_task<class k8>([=]() {}, Props); });

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k9>(
        range<1>(2), [=](id<1> index) {}, Props);
  });
  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k10>(
        range<2>(2, 2), [=](id<2> index) {}, Props);
  });
  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k11>(
        range<3>(2, 2, 2), [=](id<3> index) {}, Props);
  });
  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k12>(
        range<1>(2), id<1>(0), [=](item<1> index) {}, Props);
  });

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for<class k13>(
        ndRange, [=](nd_item<1> index) {}, Props);
  });

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for_work_group<class k14>(
        range<1>(2),
        [=](group<1> g) {
          g.parallel_for_work_item(range<1>(2), [&](h_item<1> i) {});
        },
        Props);
  });

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for_work_group<class k15>(
        range<1>(2), range<1>(2),
        [=](group<1> g) {
          g.parallel_for_work_item(range<1>(2), [&](h_item<1> i) {});
        },
        Props);
  });

  // Reductions currenty doens't work on level zero because of the gpu driver
  // issue.
  // TODO: enable when fixed.
  if (myQueue.get_context().get_platform().get_backend() !=
      backend::level_zero) {
    buffer<int, 1> InBuf1(2);
    buffer<int, 1> InBuf2(2);
    buffer<int, 1> OutBuf1(1);
    buffer<int, 1> OutBuf2(1);

    myQueue
        .submit([&](handler &CGH) {
          auto In1 = InBuf1.get_access<access::mode::read>(CGH);

          auto Out1 = OutBuf1.get_access<access::mode::discard_write>(CGH);

          auto Lambda = [=](nd_item<1> NDIt, auto &Sum1) {};

          auto Redu1 =
              ONEAPI::reduction<int, std::plus<int>>(Out1, 0, std::plus<int>{});

          auto NDR = nd_range<1>{range<1>(2), range<1>{2}};
          CGH.parallel_for<class k16>(NDR, Redu1, Lambda, Props);
        })
        .wait();

    myQueue
        .submit([&](handler &CGH) {
          auto In1 = InBuf1.get_access<access::mode::read>(CGH);
          auto In2 = InBuf2.get_access<access::mode::read>(CGH);

          auto Out1 = OutBuf1.get_access<access::mode::write>(CGH);
          auto Out2 = OutBuf2.get_access<access::mode::write>(CGH);

          auto Lambda = [=](nd_item<1> NDIt, auto &Sum1, auto &Sum2) {};

          auto Redu1 =
              ONEAPI::reduction<int, std::plus<int>>(Out1, 0, std::plus<int>{});
          auto Redu2 =
              ONEAPI::reduction<int, std::plus<int>>(Out2, 0, std::plus<int>{});

          auto NDR = nd_range<1>{range<1>(2), range<1>{2}};
          CGH.parallel_for<class k17>(NDR, Redu1, Redu2, Lambda, Props);
        })
        .wait();
  }

  return 0;
}
