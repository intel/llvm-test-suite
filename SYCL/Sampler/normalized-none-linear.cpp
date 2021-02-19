// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER

// CPU failing at moment. Waiting on fix.

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    NONE address_mode and LINEAR filter_mode

*/

#include <CL/sycl.hpp>

using namespace cl::sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::uint4;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is
// defined.
void outputPixel(sycl::uint4 somePixel) {
  std::cout << "{" << somePixel[0] << "," << somePixel[1] << "," << somePixel[2]
            << "," << somePixel[3] << "} ";
}

// some constants.

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto none = addressing_mode::none;
constexpr auto linear = filtering_mode::linear;

void test_normalized_none_linear_sampler(image_channel_order ChanOrder,
                                         image_channel_type ChanType) {
  int numTests = 7; // drives the size of the testResults buffer, and the number
                    // of report iterations. Kludge.

  // we'll use these four pixels for our image. Makes it easy to measure
  // interpolation and spot "off-by-one" probs.
  pixelT leftEdge{1, 2, 3, 4};
  pixelT body{49, 48, 47, 46};
  pixelT bony{59, 58, 57, 56};
  pixelT rightEdge{11, 12, 13, 14};

  queue Q;
  const sycl::range<1> ImgRange_1D(width);
  { // closure
    // - create an image
    image<1> image_1D(ChanOrder, ChanType, ImgRange_1D);
    event E_Setup = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::write>(cgh);
      cgh.single_task<class setupUnormLinear>([=]() {
        image_acc.write(0, leftEdge);
        image_acc.write(1, body);
        image_acc.write(2, bony);
        image_acc.write(3, rightEdge);
      });
    });
    E_Setup.wait();

    // use a buffer to report back test results.
    buffer<pixelT, 1> testResults((range<1>(numTests)));

    // sampler
    auto Norm_None_Linear_sampler = sampler(normalized, none, linear);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_norm_linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // clang-format off
        // Normalized Pixel Locations.  
        //      .125        .375        .625        .875            <-- exact center
        //  |-----^-----|-----^-----|-----^-----|-----^-----
        //[0.0         .25         .50         .75          (1)     <-- low boundary (included in pixel)
        //                                                              upper boundary inexact. (e.g. .2499999)
        // clang-format on

        // 0-2 read three pixels at inner boundary locations,  sample:
        // Normalized +  None  + Linear
        test_acc[i++] =
            image_acc.read(0.25f, Norm_None_Linear_sampler); // {25,25,25,25}
        test_acc[i++] =
            image_acc.read(0.50f, Norm_None_Linear_sampler); // {54,53,52,51}
        test_acc[i++] =
            image_acc.read(0.75f, Norm_None_Linear_sampler); // {35,35,35,35}

        // 3-6 read four pixels at exact center locations,  sample:   Normalized
        // +  None  + Linear
        test_acc[i++] =
            image_acc.read(0.125f, Norm_None_Linear_sampler); // {1,2,3,4}
        test_acc[i++] =
            image_acc.read(0.375f, Norm_None_Linear_sampler); // {49,48,47,46}
        test_acc[i++] =
            image_acc.read(0.625f, Norm_None_Linear_sampler); // {59,58,57,56}
        test_acc[i++] =
            image_acc.read(0.875f, Norm_None_Linear_sampler); // {11,12,13,14}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    auto test_acc = testResults.get_access<access::mode::read>();
    for (int i = 0, idx = 0; i < numTests; i++, idx++) {
      if (i == 0) {
        idx = 1;
        std::cout << "read three pixels at inner boundary locations,  sample:  "
                     " Normalized +  None  + Linear"
                  << std::endl;
      }
      if (i == 3) {
        idx = 0;
        std::cout << "read four pixels at exact center locations,  sample:   "
                     "Normalized +  None  + Linear"
                  << std::endl;
      }

      pixelT testPixel = test_acc[i];
      std::cout << i << " -- " << idx << ": ";
      outputPixel(testPixel);
      std::cout << std::endl;
    }
  } // ~image / ~buffer
}

int main() {

  queue Q;
  device D = Q.get_device();

  if (D.has(aspect::image)) {
    // the _int8 channels are one byte per channel, or four bytes per pixel (for
    // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
    // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
    // channel, or sixteen bytes per pixel (for RGBA).
    test_normalized_none_linear_sampler(image_channel_order::rgba,
                                        image_channel_type::unsigned_int8);
  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}

// clang-format off
// CHECK: read three pixels at inner boundary locations,  sample:   Normalized +  None  + Linear
// CHECK-NEXT: 0 -- 1: {25,25,25,25}
// CHECK-NEXT: 1 -- 2: {54,53,52,51}
// CHECK-NEXT: 2 -- 3: {35,35,35,35}
// CHECK-NEXT: read four pixels at exact center locations,  sample:   Normalized +  None  + Linear
// CHECK-NEXT: 3 -- 0: {1,2,3,4}
// CHECK-NEXT: 4 -- 1: {49,48,47,46}
// CHECK-NEXT: 5 -- 2: {59,58,57,56}
// CHECK-NEXT: 6 -- 3: {11,12,13,14}
// clang-format on
