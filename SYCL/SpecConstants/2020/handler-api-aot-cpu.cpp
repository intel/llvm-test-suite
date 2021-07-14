// REQUIRES: cpu,opencl-aot
// RUN: %clangxx -fsycl -fsycl-targets=spir64_x86_64 -Xsycl-target-backend=spir64_gen "-device *" %S/handler-api.cpp -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// FIXME: enable the test back once the segfault is fixed
// XFAIL: *
