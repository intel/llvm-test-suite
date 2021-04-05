// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fno-sycl-early-optimizations %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// The test checks that multiple calls to the same template instantiation of a
// group local memory function result in separate allocations, even with device
// code optimizations disabled (the implementation relies on inlining these
// functions regardless of device code optimization settings).

#include <CL/sycl.hpp>

#include <cassert>
#include <vector>

constexpr size_t WgSize = 32;
constexpr size_t WgCount = 4;
constexpr size_t Size = WgSize * WgCount;

class KernelA;

using namespace sycl;

int main() {
  queue Q;
  {
    std::vector<int> VecA(Size, 0);
    std::vector<int> VecB(Size, 0);
    buffer<int, 1> BufA{VecA.data(), range<1>(Size)};
    buffer<int, 1> BufB{VecB.data(), range<1>(Size)};

    Q.submit([&](handler &Cgh) {
      auto AccA = BufA.get_access<access::mode::read_write>(Cgh);
      auto AccB = BufB.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<KernelA>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            multi_ptr<int[WgSize], access::address_space::local_space> PtrA =
                group_local_memory_for_overwrite<int[WgSize]>(Item.get_group());
            multi_ptr<int[WgSize], access::address_space::local_space> PtrB =
                group_local_memory_for_overwrite<int[WgSize]>(Item.get_group());
            size_t LocalId = Item.get_local_linear_id();
            (*PtrA)[LocalId] = 1;
            (*PtrB)[LocalId] = 1;
            (*PtrA)[LocalId] *= LocalId;
            (*PtrB)[LocalId] *= LocalId;
            Item.barrier();

            size_t LocalIdx = LocalId ^ 1;
            size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
            AccA[GlobalIdx] = (*PtrA)[LocalIdx];
            AccB[GlobalIdx] = (*PtrB)[LocalIdx];
          });
    });

    auto AccA = BufA.get_access<access::mode::read>();
    auto AccB = BufB.get_access<access::mode::read>();
    for (size_t I = 0; I < Size; ++I) {
      assert(AccA[I] == I % WgSize);
      assert(AccB[I] == I % WgSize);
    }
  }
}
