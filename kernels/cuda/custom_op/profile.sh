pkill -f "test_ar_graph.py --world-size 8"
pkill -f "test_ar.py --world-size 8"
pkill -f "test_rs_fused.py"

export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

nsys profile --trace=cublas,cuda,cudnn,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true python test_ar_graph.py --world-size 8

nsys launch --trace=cublas,cuda,cudnn,nvtx --cuda-graph-trace node --show-output=true --session-new test_ar_graph \
    python test_ar_graph.py --world-size 8

nsys start --output /cpfs/user/baiheng/code/play_kernel/play_kernel/kernels/cuda/custom_op/test_ar_graph.nsys-rep --stats=true  --session test_ar_graph
nsys stop test_ar_graph