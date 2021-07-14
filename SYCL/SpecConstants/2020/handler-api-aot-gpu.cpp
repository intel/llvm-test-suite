// REQUIRES: gpu,ocloc
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" %S/handler-api.cpp -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
