// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==---------- buffer_dev_to_dev.cpp - SYCL buffer basic test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <memory>

using namespace cl::sycl;

int main() {
  int Data[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  {

    const std::vector<platform> Platforms = platform::get_platforms();

    if (Platforms.size() < 2) {
      std::cout << "Need at least two platforms to create two separate "
                   "contexts. Skipping...\n";
      return 0;
    }

    std::vector<queue> Queues;

    for (const platform &Platform : Platforms) {
      const std::vector<device> Devices = Platform.get_devices();
      assert(!Devices.empty() && "No devices in the platform?");

      Queues.emplace_back(Devices[0]);
    }

    buffer<int, 1> Buffer(Data, range<1>(10),
                          {property::buffer::use_host_ptr()});

    assert(Queues[0].get_context() != Queues[1].get_context());
    Queues[0].submit([&](handler &Cgh) {
      auto Accessor = Buffer.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<class init_b>(range<1>{10},
                                     [=](id<1> Index) { Accessor[Index] = 0; });
    });
    Queues[1].submit([&](handler &Cgh) {
      auto Accessor = Buffer.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<class increment_b>(
          range<1>{10}, [=](id<1> Index) { Accessor[Index] += 1; });
    });
  } // Data is copied back
  for (int I = 0; I < 10; I++) {
    assert(Data[I] == 1);
  }

  return 0;
}
