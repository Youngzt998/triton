
# A manual script to build triton (if you wish to control every step by yourself)

# Basic logic to build Triton from scratch:
# 1. build LLVM with MLRI enabled
# 2. setup python virtual env locally
# 3. build Triton with the customized LLVM

# Step 0: clone the repo of triton and llvm-project (could be from opensource or internal)


# Step 1: Build LLVM with default cmake settings

# note on a new machine, it will requires clang, lld, cmake, ninja-build installed

# e.g. our devservers are usually CentOS, we run "sudo dnf install xxx" to install missing packages, i.e.
sudo dnf install python3.11-devel python3.11 ccache cmake ninja-build clang llvm lld zlib zlib-devl

# then configure the LLVM build options (usually enabling debug) with cmake:
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_CCACHE_BUILD=OFF \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
  -DCMAKE_INSTALL_PREFIX= `pwd`/destdir \
  -B`pwd`/build `pwd`/llvm
# and build or rebuild after chaning it at llvm's repo root (e.g. ./llvm-project)
ninja -C build

# Note: python version 3.11 or 3.12 recommended

# set up python venv without conda: create a virtual env folder named .venv at current path
python3 -m venv .venv --prompt triton #
# active the virtual env by
source .venv/bin/activate
# deactive the virtual env by
deactive

# we may also create as much ven as we want for different purpose
python3 -m venv .venv-nvptx --prompt triton
source .venv-nvptx/bin/activate

# install dependencies inside virtual env if not there
pip3 install ninja cmake wheel scipy numpy pytest pytest-xdist pytest-forked lit pandas matplotlib pybind11 expecttest hypothesis pre-commit

# install torch for amd gpu:
export CUDA_OR_ROCM=rocm
export CUDA_MAJOR=6
export CUDA_MINOR=".3"
export TORCH_URL=https://download.pytorch.org/whl/nightly/$CUDA_OR_ROCM$CUDA_MAJOR$CUDA_MINOR # pip install that get us torch 2.7.1 won't work, since triton is newer
pip3 install --no-cache-dir --pre torch torchvision torchaudio --index-url $TORCH_URL

# or for nvidia gpu:
export CUDA_OR_ROCM=cu
export CUDA_MAJOR=12
export CUDA_MINOR=6
export CUDA_HOME=/usr/local/cuda-$CUDA_MAJOR.$CUDA_MINOR
export PATH=$CUDA_HOME/bin:$PATH
export USE_CUDA=1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export TORCH_URL=https://download.pytorch.org/whl/nightly/$CUDA_OR_ROCM$CUDA_MAJOR$CUDA_MINOR
pip3 install --no-cache-dir --pre torch torchvision torchaudio --index-url $TORCH_URL


# TRAP: the pip install above may install a official lib of triton, which makes your environment have 2 version of triton after install
#   your program may use the default one instead of your own build, and you need to delete it by hand if that happens


# After running the above, to build & rebuild Triton, run:
LLVM_BUILD_DIR=`pwd`/../triton-llvm/build \
DEBUG=1 \
TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=0 \
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
LLVM_SYSPATH=$LLVM_BUILD_DIR \
    pip3 install -e . --no-build-isolation


# run the tutorial
python3 python/tutorials/01-vector-add.py

## Cheatsheet on debugging triton compiler:

# run python only with pdb breakpoint:
# insert one line of "breakporint()" in python codes, and run it

# run python-cpp whole thing in debugging mode with lldb
SOME_EVN_OPTION=1 lldb -- python3 python/tutorials/01-vector-add.py

# Important optional flags
TRITON_CACHE_DIR=./.dump \  # dump all the ir into ./.dump
TRITON_ALWAYS_COMPILE=1 \   # make sure triton always compile the code when running, instead of using compiled kernel in cache
python python/tutorials/01-vector-add.py

# Other important flags:

# MLIR_DUMP_PATH=./.dump
# TRITON_KERNEL_DUMP=1
# TRITON_DUMP_DIR=./.dump
# MLIR_ENABLE_DUMP=1


# run triton-opt on the dumped ir
./triton/build/xxxxxxx/bin/triton-opt \
   --mlir-print-debuginfo \         # this ensure location attribute is printed
   --mlir-use-nameloc-as-prefix \   # this ensure a named location's name will be used as corresponding value's name
   xxx.ttir

# or in debugger
lldb -- ./triton/build/cmake.xxxx/bin/triton-opt xxx.ttir


# we can also run triton-opt on different ir files:
./build/cmake.xxxxxxxxxxxx/bin/triton-opt xxx.source
./build/cmake.xxxxxxxxxxxx/bin/triton-opt xxx.ttir
./build/cmake.xxxxxxxxxxxx/bin/triton-opt xxx.ttgir


# running lit tests
lit -sv test/Triton/loop_cse.mlir

# run all lit tests of the project
make test-lit

# run a single FileCheck test
/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt \
  /data/users/youngzt/triton-osmeta-name-preservation/triton/test/Triton/add_kernel.ttir \
  -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix \
  | /data/users/youngzt/triton-osmeta-name-preservation/triton-llvm/build/bin/FileCheck \
    /data/users/youngzt/triton-osmeta-name-preservation/triton/test/Triton/add_kernel.ttir

# supposing we registerd my own new pass "triton-ttir-testhelloworld", and wish to using triton-opt to dumping out debugging info, we can do:
# the flag will be automatically registered
triton-opt -triton-ttir-testhelloworld xxx.ttir
# or using built-in flags
triton-opt --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info  ~/dbg_add_kernel.llvmir
triton-opt --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info ~/add_kernel.llvmir > ~/add_kervel.llvmir

# run python tests
python -m pytest -vvv



# convert a mlir's llvm dialect to llvm ir
mlir-translate --mlir-to-llvmir ~/dbg_add_kernel.llvmir



# !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)

# Generate a ptx file
../triton-llvm/build/bin/llc -march=nvptx64 -mcpu=sm_90 --debug-entry-values dbg_add_kernel.ll -o dbg_add_kernel.ptx
./python/triton/backends/nvidia/bin/ptxas -arch=sm_90a ~/add_kernel.ptx -o ~/add_kernel.cubin

# submit your commit as diff after git commit:
jf submit
# or a stacked diffs
jf submit --stack

# address code review comments by rebasing and amending
git rebase -i HEAD~$A_NUMBER
git commit -a --amend --no-edit
jf submit --stack

# store a new diff locally without git commit
git show HaShTaGxxxxxxxx > patch_name.patch

# mlir::Type resultType = op->getResult(0).getType();

# exam dwarf debug info in an object file (e.g. cubin)
../triton-llvm/build/bin/llvm-dwarfdump softmax_kernel.cubin

# ../triton-llvm/build/bin/llvm-dwarfdump softmax_kernel.cubin | grep name | sed 's/.*(\"//g' | sed 's/\")//g' | sort | uniq  | grep -v pubname | grep -v -E "float|vector|int|bool|type|softmax"
