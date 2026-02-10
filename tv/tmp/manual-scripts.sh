

# build triton-tv only
ninja -C /home/zyang/triton/build/cmake.linux-x86_64-cpython-3.12 triton-tv

# run triton-tv on two (required) mlir files
build/cmake.linux-x86_64-cpython-3.12/tv/triton-tv ./tv/tmp/file1 ./tv/tmp/file2