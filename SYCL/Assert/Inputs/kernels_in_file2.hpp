#include <CL/sycl.hpp>

using namespace cl::sycl;

SYCL_EXTERNAL int calculus(int X);

void enqueueKernel_1_fromFile2(queue *Q);

void enqueueKernel_2_fromFile2(queue *Q);
