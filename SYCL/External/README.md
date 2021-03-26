# Overview

"External" tests directory has necessary infrastructure for running tests where
sources of tests are expected to be located outside of llvm-test-suite
repository. Each subdirectory represents external repository for corresponding
application/test.

# Directory structure
Every sub-directory (e.g. RSBench) contains the following content:
 * **CMakeLists.txt** - CMake configuration file which is used to obtain
 application binary and data files.
 * **\*.test** - test configuration files, containing command lines, test
   status lines and verification patterns following
   [LLVM tests infrastructure](https://llvm.org/docs/TestingGuide.html).
 * **lit.local.cfg** - application specific LIT infrastructure configuration
   file. In case of RSBench benchmarks it sets LIT test file suffixes to `.test`
   and mark all tests in directory unsupported if corresponding directory is
   not added to `SYCL_EXTERNAL_TESTS` list.

# CMake parameters

All parameters described in [Readme.md](../README.md#cmake-parameters) are
applicable.

***SYCL_EXTERNAL_TESTS*** semicolon-separated names of external SYCL
applications which are built and run as part of the testing. Name is
subdirectory name in "External" directory. Example:
```
-DSYCL_EXTERNAL_TESTS=RSBench
```
Source code of external application can be downloaded from external repository
as part of the build or provided in CMake variable <APPNAME>_SRC
(e.g. RSBench_SRC).

Also extra CMake parameters are introduced to configure specific
application:
   * **APPName_BIN** (e.g. `RSBench_BIN`) - directory containing prebuilt
     binaries of the application.
   * **APPName_SRC** (e.g. `RSBench_SRC`) - directory containing sources of the
     application.
   * **APPName_URL** (e.g. `RSBench_URL`) - URL to the GIT repository containing
     sources of the application.
   * **APPName_TAG** (e.g. `RSBench_TAG`) - GIT tag or hash or branch name used
     to download source from GIT repository.

Configuration parameters are priorities from top to down. If **APPName_BIN**
is specified binaries will be used directly ignoring other parameters.

# Build and run tests

Get sources

```
git clone https://github.com/intel/llvm-test-suite
cd llvm-test-suite
mkdir build
cd build
```

With compiler tools available in the PATH:

```
# Configure
cmake \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DTEST_SUITE_SUBDIRS=SYCL \
 -DSYCL_TARGET_DEVICES="level_zero:gpu" \
 -DSYCL_EXTERNAL_TESTS="RSBench" \
 ..

# Build and Run
make check-sycl-all

```

