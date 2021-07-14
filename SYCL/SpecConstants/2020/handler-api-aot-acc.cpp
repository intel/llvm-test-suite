// REQUIRES: accelerator,aoc

// RUN: %clangxx -fsycl -fsycl-targets=spir64_fpga %S/handler-api.cpp -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
