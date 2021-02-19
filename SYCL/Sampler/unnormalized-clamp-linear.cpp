// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out %HOST_CHECK_PLACEHOLDER


// GPU does not correctly interpolate when using clamp.  Waiting on fix. Both OCL and LevelZero have this issue.
// CPU failing all linear interpolation at moment. Waiting on fix.


/*
    This file sets up an image, initializes it with data, 
    and verifies that the data is sampled correctly with a
    sampler configured UNNORMALIZED coordinate_normalization_mode
    CLAMP address_mode and LINEAR filter_mode
*/

#include <CL/sycl.hpp>

using namespace cl::sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::uint4;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is defined.
void outputPixel(sycl::uint4 somePixel){
    std::cout << "{" << somePixel[0]
              << "," << somePixel[1]
              << "," << somePixel[2]
              << "," << somePixel[3]
              << "} ";
}


// some constants.

// 4 pixels on a side. 1D at the moment
constexpr long width    = 4;

constexpr auto unnormalized = coordinate_normalization_mode::unnormalized;
constexpr auto linear       = filtering_mode::linear;



void test_unnormalized_clamp_linear_sampler(image_channel_order ChanOrder, image_channel_type ChanType){
    int numTests = 8; // drives the size of the testResults buffer, and the number of report iterations. Kludge.


    // we'll use these four pixels for our image. Makes it easy to measure interpolation and spot "off-by-one" probs.
    pixelT leftEdge{1,2,3,4};
    pixelT body{49,48,47,46};
    pixelT bony{59,58,57,56};
    pixelT rightEdge{11,12,13,14};

    queue Q;
    const sycl::range<1> ImgRange_1D(width);
    { // closure
        // - create an image
        image<1> image_1D(ChanOrder, ChanType, ImgRange_1D);
        event E_Setup = Q.submit([&](handler &cgh){
            auto image_acc = image_1D.get_access<pixelT, access::mode::write>(cgh);
            cgh.single_task<class setupUnormLinear>([=] () {
                image_acc.write(0, leftEdge);
                image_acc.write(1, body);
                image_acc.write(2, bony);
                image_acc.write(3, rightEdge);
            });
        });
        E_Setup.wait();

        //use a buffer to report back test results.  
        buffer<pixelT,1> testResults((range<1>(numTests)));

        //sampler
        auto UnNorm_Clamp_Linear_sampler      = sampler(unnormalized, addressing_mode::clamp, linear);

        event E_Test = Q.submit([&](handler &cgh){
            auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
            auto test_acc = testResults.get_access<access::mode::write>(cgh);

            cgh.single_task<class im1D_Unorm_Linear>([=] () {
                int i = 0; //the index for writing into the testResult buffer. 

                // UnNormalized Pixel Locations when using Linear Interpolation
                //(0 -------- ](1 ---------- ](2 ----------- ](3---------- ](4)

                
                //0-5 read six pixels, float coordinates,   sample:   NonNormalized + Clamp + Linear
                test_acc[i++] = image_acc.read(-1.0f, UnNorm_Clamp_Linear_sampler); // {0,0,0,0}
                test_acc[i++] = image_acc.read(0.0f, UnNorm_Clamp_Linear_sampler);  // {0,1,2,2}  
                test_acc[i++] = image_acc.read(1.0f, UnNorm_Clamp_Linear_sampler);  // {25,25,25,25}
                test_acc[i++] = image_acc.read(2.0f, UnNorm_Clamp_Linear_sampler);  // {54,53,52,51}
                test_acc[i++] = image_acc.read(3.0f, UnNorm_Clamp_Linear_sampler);  // {35,35,35,35}
                test_acc[i++] = image_acc.read(4.0f, UnNorm_Clamp_Linear_sampler);  // {6,6,6,7}  // interpolated?

                //6-7 read two pixels on either side of first pixel. float coordinates. CLAMP
                //  on GPU CLAMP is apparently stopping the interpolation. ( values on right are expected value)
                test_acc[i++] = image_acc.read(0.9999f, UnNorm_Clamp_Linear_sampler); // {25,25,25,25}
                test_acc[i++] = image_acc.read(1.0001f, UnNorm_Clamp_Linear_sampler); // {25,25,25,25}

            });
        });
        E_Test.wait();

        // REPORT RESULTS
        auto test_acc = testResults.get_access<access::mode::read>();
        for(int i = 0, idx=0; i < numTests; i++, idx++){
            if(i==0){ idx=-1; std::cout << "read six pixels, float coordinates,   sample:   NonNormalized + Clamp + Linear" << std::endl; }
            if(i==6){ idx=1; std::cout << "read two pixels on either side of 1. float coordinates. NonNormalized + Clamp + Linear" << std::endl; }
            if(i==7){ idx=1;  }

            
            pixelT testPixel = test_acc[i];
            std::cout << i <<  " -- " << idx <<  ": "; 
            outputPixel(testPixel);
            std::cout << std::endl;
        }
    } // ~image / ~buffer   
}

int main(){

    queue Q;
    device D = Q.get_device();

    if(D.has(aspect::image)){ 
        // the _int8 channels are one byte per channel, or four bytes per pixel (for RGBA)
        // the _int16/fp16 channels are two bytes per channel, or eight bytes per pixel (for RGBA)
        // the _int32/fp32  channels are four bytes per channel, or sixteen bytes per pixel (for RGBA).
        test_unnormalized_clamp_linear_sampler(image_channel_order::rgba, image_channel_type::unsigned_int8);
    } else {
        std::cout << "device does not support image operations"  << std::endl;
    }


    return 0;
}

//CHECK: read six pixels, float coordinates,   sample:   NonNormalized + Clamp + Linear
//CHECK-NEXT: 0 -- -1: {0,0,0,0} 
//CHECK-NEXT: 1 -- 0: {0,1,2,2} 
//CHECK-NEXT: 2 -- 1: {25,25,25,25} 
//CHECK-NEXT: 3 -- 2: {54,53,52,51} 
//CHECK-NEXT: 4 -- 3: {35,35,35,35} 
//CHECK-NEXT: 5 -- 4: {6,6,6,7} 
//CHECK-NEXT: read two pixels on either side of 1. float coordinates. NonNormalized + Clamp + Linear
//CHECK-NEXT: 6 -- 1: {25,25,25,25} 
//CHECK-NEXT: 7 -- 1: {25,25,25,25} 
