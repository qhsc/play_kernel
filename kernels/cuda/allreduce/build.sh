#!/bin/bash

# 构建脚本 - 编译 ar.cu 为共享库
set -e

echo "开始构建 sgl_kernel_allreduce..."

# 创建构建目录
mkdir -p build
cd build

# 配置CMake (支持自定义路径)
echo "配置CMake..."
if [ -n "$CUDA_ROOT" ]; then
    echo "使用自定义CUDA路径: $CUDA_ROOT"
fi
if [ -n "$TORCH_ROOT" ]; then
    echo "使用自定义PyTorch路径: $TORCH_ROOT"
fi

if [ -n "$TORCH_ROOT" ]; then
    cmake .. -DCUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda} -DTORCH_ROOT=${TORCH_ROOT}
else
    cmake .. -DCUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
fi

# 构建项目
echo "开始编译..."
make -j$(nproc)

echo "构建完成！"