cmake_minimum_required(VERSION 3.20)

project(toy-lang)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Simple approach: just include LLVM headers and link basic libraries
include_directories(/usr/include/llvm-20)
include_directories(/usr/include/llvm-c-20)

# Add library search path
link_directories(/usr/lib/llvm-20/lib)

add_executable(toy-lang src/lang.cc src/codegen.cc)

# Link basic LLVM libraries
target_link_libraries(toy-lang LLVMCore LLVMSupport)