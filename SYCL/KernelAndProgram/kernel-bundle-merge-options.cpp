// RUN: %clangxx -fsycl %s -o %t.out -g
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 %t.out %GPU_CHECK_PLACEHOLDER
// REQUIRES: gpu
// UNSUPPORTED: cuda
#include "kernel-bundle-merge-options.hpp"

// CHECK: piProgramBuild
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <const char *>: -g -vc-codegen

// TODO: Uncomment when build options are properly passed to compile and link
//       commands for kernel_bundle
// xCHECK: piProgramCompile(
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <const char *>: -g -vc-codegen
// xCHECK: piProgramLink(
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <const char *>: -g -vc-codegen
