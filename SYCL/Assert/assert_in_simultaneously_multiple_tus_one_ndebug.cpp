// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DDEFINE_NDEBUG_INFILE2 -I %S/Inputs %S/assert_in_simultaneously_multiple_tus.cpp %S/Inputs/kernels_in_file2.cpp -o %t.out %threads_lib
// RUN: %CPU_RUN_PLACEHOLDER %t.out | %CPU_RUN_PLACEHOLDER FileCheck %s
// RUN: %GPU_RUN_PLACEHOLDER %t.out | %GPU_RUN_PLACEHOLDER FileCheck %s
// RUN: %ACC_RUN_PLACEHOLDER %t.out | %ACC_RUN_PLACEHOLDER FileCheck %s
//
// CHECK:      this message from file1
// CHECK-NOT:  this message from file2
// CHECK-NOT:  The test ended.
