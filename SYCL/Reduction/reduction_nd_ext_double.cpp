// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DTEST_SYCL2020_REDUCTIONS %s -o %t2020.out
// RUN: %CPU_RUN_PLACEHOLDER %t2020.out
// RUN: %GPU_RUN_PLACEHOLDER %t2020.out
// RUN: %ACC_RUN_PLACEHOLDER %t2020.out

// TODO: Enable the test for HOST when it supports intel::reduce() and barrier()
// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// RUNx: %HOST_RUN_PLACEHOLDER %t2020.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// used with 'double' type.

#include "reduction_nd_ext_type.hpp"

int main() { return runTests<double>("cl_khr_double"); }
