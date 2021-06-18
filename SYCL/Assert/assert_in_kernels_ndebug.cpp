// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DNDEBUG %S/assert_in_kernels.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out | %CPU_RUN_PLACEHOLDER FileCheck %s
// RUN: %GPU_RUN_PLACEHOLDER %t.out | %GPU_RUN_PLACEHOLDER FileCheck %s
// RUN: %ACC_RUN_PLACEHOLDER %t.out | %ACC_RUN_PLACEHOLDER FileCheck %s
//
// CHECK-NOT: One shouldn't see this message
// CHECK-NOT: from assert statement
// CHECK-NOT: test aborts earlier, one shouldn't see this message
// CHECK: The test ended.
