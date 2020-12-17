//==--------------- tpm_pointer_v2.cpp - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out 1
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out 2
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out 3

// Since in ESIMD a single WI occupies entire underlying H/W thread, SYCL
// private memory maps to what's known as 'thread private memory' in CM.
// This test is intended to use TPM to support implementation in ESIMD
// backend. In order to force using of TPM need to allocate 96x32 bytes or more.

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

int main(int argc, char **argv) {
  constexpr unsigned VL = 8;
  constexpr unsigned SZ = 800; // big enough to use TPM

  if (argc != 2) {
    std::cout << "Skipped! Specify case number" << std::endl;
    return 1;
  }

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctx = q.get_context();

  int *output =
      static_cast<int *>(sycl::malloc_shared(VL * sizeof(int), dev, ctx));
  memset(output, 0, VL * sizeof(int));

  int case_num = atoi(argv[1]);
  std::cout << "CASE NUM: " << case_num << std::endl;

  int offx1 = 111;
  int offx2 = 55;
  int offy1 = 499;
  int offy2 = 223;
  int offz = 99;
  int base1 = 500;
  int base2 = 100;
  int divisor = 4;

  {
    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Test>(
          sycl::range<1>{1}, [=](id<1> i) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;
            simd<int, VL> val(0);

            int x1[SZ];
            for (int j = 0; j < SZ; ++j) {
              int idx = (j + offx1) % SZ;
              x1[idx] = (idx % 2) == 0 ? j : base1;
            }

            int x2[SZ];
            for (int j = 0; j < SZ; ++j) {
              int idx = (j + offx2) % SZ;
              x2[idx] = base2 << (j % 32);
            }

            // some work with X1
            for (int j = 1; j < SZ; ++j) {
              if ((x1[j] + j) > base1)
                x1[j] = (j * (x1[j] + x1[j - 1]) / divisor) - base2;
            }

            // some work with X2
            for (int j = 1; j < SZ; ++j) {
              if ((x2[j] + j) < base2)
                x2[j] = (divisor * (x2[j] - x2[j - 1]) / j) + base1;
            }

            if (case_num == 1) {
              for (int j = 0; j < SZ; ++j)
                val.select<1, 1>(j % VL) += x1[j] - x2[j];
            } else {
              int *y1[SZ];
              for (int j = 0; j < SZ; ++j) {
                int idx = (j + offy1) % SZ;
                y1[j] = j % 6 == 0 ? x1 + idx : x2 + idx;
              }

              int *y2[SZ];
              for (int j = 0; j < SZ; ++j) {
                int idx = (j + offy2) % SZ;
                y2[j] = j % 2 == 0 ? x2 + idx : x1 + idx;
              }

              // some work with Y1
              for (int j = 0; j < SZ; j += 2) {
                if (*(y1[j]) > *(y1[j + 1]))
                  *(y1[j]) = *(y1[j + 1]) - *(y1[j]);
              }

              // some work with Y2
              for (int j = 1; j < SZ - 1; j += 2) {
                if ((*(y2[j]) <= *(y2[j + 1]))) {
                  auto temp = y2[j];
                  y2[j] = y2[j + 1];
                  y2[j + 1] = temp;
                }
              }

              if (case_num == 2) {
                for (int j = 0; j < SZ; ++j)
                  val.select<1, 1>(j % VL) += *(y1[j]) - *(y2[j]);
              } else { // case_num == 3
                int **z[SZ];
                for (int j = 0; j < SZ; ++j) {
                  int idx = (j + offz) % SZ;
                  z[j] = y1 + idx;
                }

                // some work with Z
                for (int j = 0; j < SZ - 1; ++j) {
                  if (*(*(z[j])) < *(*(z[j + 1])))
                    z[j] = y2 + j;
                  if (j % 18 == 0)
                    (*(*(z[j])))++;
                }

                for (int j = 0; j < SZ; ++j)
                  val.select<1, 1>(j % VL) += *(*(z[j]));
              }
            }

            block_store<int, VL>(output, val);
          });
    });
    e.wait();
  }

  int o[VL] = {0};

  int x1[SZ];
  for (int j = 0; j < SZ; ++j) {
    int idx = (j + offx1) % SZ;
    x1[idx] = (idx % 2) == 0 ? j : base1;
  }

  int x2[SZ];
  for (int j = 0; j < SZ; ++j) {
    int idx = (j + offx2) % SZ;
    x2[idx] = base2 << (j % 32);
  }

  // some work with X1
  for (int j = 1; j < SZ; ++j) {
    if ((x1[j] + j) > base1)
      x1[j] = (j * (x1[j] + x1[j - 1]) / divisor) - base2;
  }

  // some work with X2
  for (int j = 1; j < SZ; ++j) {
    if ((x2[j] + j) < base2)
      x2[j] = (divisor * (x2[j] - x2[j - 1]) / j) + base1;
  }

  if (case_num == 1) {
    for (int j = 0; j < SZ; ++j)
      o[j % VL] += x1[j] - x2[j];
  } else {
    int *y1[SZ];
    for (int j = 0; j < SZ; ++j) {
      int idx = (j + offy1) % SZ;
      y1[j] = j % 6 == 0 ? x1 + idx : x2 + idx;
    }

    int *y2[SZ];
    for (int j = 0; j < SZ; ++j) {
      int idx = (j + offy2) % SZ;
      y2[j] = j % 2 == 0 ? x2 + idx : x1 + idx;
    }

    // some work with Y1
    for (int j = 0; j < SZ; j += 2) {
      if (*(y1[j]) > *(y1[j + 1]))
        *(y1[j]) = *(y1[j + 1]) - *(y1[j]);
    }

    // some work with Y2
    for (int j = 1; j < SZ - 1; j += 2) {
      if ((*(y2[j]) <= *(y2[j + 1]))) {
        auto temp = y2[j];
        y2[j] = y2[j + 1];
        y2[j + 1] = temp;
      }
    }

    if (case_num == 2) {
      for (int j = 0; j < SZ; ++j)
        o[j % VL] += *(y1[j]) - *(y2[j]);
    } else { // case_num == 3
      int **z[SZ];
      for (int j = 0; j < SZ; ++j) {
        int idx = (j + offz) % SZ;
        z[j] = y1 + idx;
      }

      // some work with Z
      for (int j = 0; j < SZ - 1; ++j) {
        if (*(*(z[j])) < *(*(z[j + 1])))
          z[j] = y2 + j;
        if (j % 18 == 0)
          (*(*(z[j])))++;
      }

      for (int j = 0; j < SZ; ++j)
        o[j % VL] += *(*(z[j]));
    }
  }

  int err_cnt = 0;
  for (int j = 0; j < VL; ++j)
    if (output[j] != o[j])
      err_cnt += 1;

  sycl::free(output, ctx);

  if (err_cnt > 0) {
    std::cout << "FAILED.\n";
    return 1;
  }

  std::cout << "Passed\n";
  return 0;
}
