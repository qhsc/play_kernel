cmake_minimum_required(VERSION 3.18)
project(sgl_kernel_allreduce)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90)

# 设置CUDA和PyTorch路径 (可以通过环境变量或命令行参数指定)
if(NOT DEFINED CUDA_ROOT)
    set(CUDA_ROOT "/usr/local/cuda")
endif()
if(NOT DEFINED TORCH_ROOT)
    # 尝试从Python环境自动检测PyTorch路径
    execute_process(
        COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYTHON_RESULT
    )
    if(NOT PYTHON_RESULT EQUAL 0)
        message(FATAL_ERROR "无法找到PyTorch，请设置TORCH_ROOT环境变量或安装PyTorch")
    endif()
endif()

# 设置路径
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
set(CMAKE_PREFIX_PATH ${TORCH_ROOT} ${CMAKE_PREFIX_PATH})

# 查找PyTorch
find_package(Torch REQUIRED)

# 源文件
set(SOURCES src/custom_all_reduce.cu)

# 创建共享库
add_library(my_ar SHARED ${SOURCES})

# 设置目标属性
set_target_properties(my_ar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 链接库
target_link_libraries(my_ar ${TORCH_LIBRARIES} cudart cuda)

# 包含目录
target_include_directories(my_ar PRIVATE src)

# 编译选项
target_compile_options(my_ar PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
)

# 清理PyTorch的编译选项
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")

# 输出信息
message(STATUS "CUDA路径: ${CUDA_ROOT}")
message(STATUS "PyTorch路径: ${TORCH_ROOT}")
message(STATUS "PyTorch版本: ${Torch_VERSION}")
