# SGL Kernel AllReduce

基于CUDA的多GPU All-Reduce实现，可以编译为共享库。

## 文件结构

- `src/ar.cu` - 主CUDA源文件
- `src/ar.cuh` - 头文件，包含CUDA内核和数据结构
- `CMakeLists.txt` - CMake配置文件
- `build.sh` - 构建脚本

## 依赖

- CUDA 12.6+ 
- PyTorch (通过Python环境自动检测)
- CMake 3.18+
- C++17编译器

## 构建

### 自动构建
```bash
./build.sh
```

### 手动构建
```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### 指定自定义路径
```bash
# 指定CUDA路径
CUDA_ROOT=/path/to/cuda ./build.sh

# 指定PyTorch路径
TORCH_ROOT=/path/to/torch ./build.sh

# 同时指定
CUDA_ROOT=/path/to/cuda TORCH_ROOT=/path/to/torch ./build.sh
```

## 输出

构建完成后，共享库位于：
`build/lib/libsgl_kernel_allreduce.so`

## 特性

- 自动从Python环境检测PyTorch路径
- 支持自定义CUDA和PyTorch路径
- 优化的CUDA编译选项
- 支持CUDA架构90 (H100等)
- 简洁的CMake配置

## 使用

共享库导出 `init_custom_ar` 函数，用于初始化多GPU自定义All-Reduce操作。
