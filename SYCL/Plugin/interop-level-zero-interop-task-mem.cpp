// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test for Level Zero interop_task.

#include <CL/sycl.hpp>
// clang-format off
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>
// clang-format on

#define SIZE 16

class my_selector : public cl::sycl::device_selector {
public:
  int operator()(const cl::sycl::device &dev) const override {
    sycl::backend backend = dev.get_platform().get_backend();
    if (backend == cl::sycl::backend::level_zero && dev.is_gpu())
      return 1;
    else
      return 0;
  }
};

int main() {
  sycl::queue queue = sycl::queue(my_selector());

  ze_context_handle_t ze_context =
      queue.get_context().get_native<sycl::backend::level_zero>();

  try {
    sycl::buffer<uint8_t, 1> buffer(SIZE);
    sycl::image<2> image(sycl::image_channel_order::rgba,
                         sycl::image_channel_type::fp32, {SIZE, SIZE});

    queue
        .submit([&](cl::sycl::handler &cgh) {
          auto buffer_acc =
              buffer.get_access<cl::sycl::access::mode::write>(cgh);
          auto image_acc =
              image.get_access<sycl::float4, sycl::access::mode::write>(cgh);
          cgh.interop_task([&](const cl::sycl::interop_handler &ih) {
            void *device_ptr =
                ih.get_mem<sycl::backend::level_zero>(buffer_acc);
            size_t size = 0;
            zeMemGetAddressRange(ze_context, device_ptr, NULL, &size);
            assert(size == SIZE);

            ze_image_handle_t ze_image =
                ih.get_mem<sycl::backend::level_zero>(image_acc);
            assert(ze_image != nullptr);
          });
        })
        .wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << std::endl;
    return e.get_cl_code();
  } catch (const char *msg) {
    std::cout << "Exception caught: " << msg << std::endl;
    return 1;
  }

  return 0;
}
