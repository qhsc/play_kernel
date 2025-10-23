# FA1
## 算法
* 输入 `Q,K,V in [N,d]`，Shared Memory 大小 M
* 在 HBM 上分配 `O in [N,d], l in [N]` 都初始化成 `0`，`m in [N]` 初始化成`-inf`，`l` 用来存储 sum(exp), `m` 存储最大值。
* 计算 TileSize for Q and K,V
    * `Bc = M/d/4`, `Br = min(M/d/4, d)`
    * `Q O l m` 都划分成 `Tr` 个，`K V` 拆分成 `Tc` 个
* `for 1 <= j <= Tc`
    * load `Kj, Vj`
    * `for 1 <= i <= Tr`
        * load `Qi Oi li mi`
        * On-chip 计算当前 block 的值
            * `Sij = Qi @ Kj^T in [Br,Bc]`
            * `mij = rowmax(Sij) in [Br]`
            * `Pij = exp(Sij-mi)in [Br,Bc]`
            * `lij = rowsum(Pij) in [Br,Bc]`
        * On-chip 计算更新值
            * `mi_new = max(mij, mi)`
            * `li_new = li*exp(mi-mi_new) + lij*exp(mij-mi_new)`
        * 计算 Oi 的值并写入 HBM
            * `Oi = Pij @ Vj^T * exp(mij-mi_new) / li_new + li/li_new * exp(mi-mi_new)*O_i`
        * 更新 l 和 m 到 HBM
            * `mi <- mi_new, li <- li_new`
    * `end row Tr for`
* `end column Tc for`

# FA2
相比于 FA1，FA2 的提升有：
* 交换了两个 loop 的顺序
    * 可以在 Q 方向切分做并行计算，且不同的 Q block 之间没有数据依赖。在 FA1 中，如果在 KV 方向切分做并行计算，则不同的 block 之间是有数据依赖的。提供了更好的并行性，特别是在 Decode 这种 Bs 不大，但是 Sequence 很长的场景中十分适用。这个改变在 CTA level 和 Warp level 都发生了。
    * 对于 inference 来说不再需要 HBM 上的 `l, m` 了，这两个可以直接放到 Shared Memory 上面。这样FA1 中更新的 epilog 就减少了。
* CUDA Core 计算量降低：避免每次更新 O 都计算缩放。改成内层循环做完后才做一次缩放。

## 算法
* 输入 `Q K V in [N,d]`, `Br, Bc`
* 初始化 `O in [N,d]`
* `Q O` 按照 `Br` 拆分成 `Tr` 份，`K V` 按照 `Bc` 拆分成 `Tc` 份
* ` for i in range(Tr)`
    * load `Qi`
    * On-chip 初始化 `Oi in [Br,d], l in [Br], m in [Br]`
    * `for j in range(Tc)`
        * load `Kj, Vj`
        * on-chip compute:
            * `Sij = Qi @ Kj^T`
            * `m_old = rowmax(Sij)`
            * `m = max(m, m_old)`
            * `Pij = exp(Sij-m)`
            * `l = exp(m_old-m)*l + rowsum(Pij)`
            * `O = O*exp(m_old-m) Pij@V`
    *  `end for KV`
    * `O_i HBM <- O/l`
* `end for Q`

# FA3
相对于 FA2，FA3引入了针对 Hopper 架构的 TMA 特性来优化数据加载，并且支持 FP8 attention。因为使用了 TMA 了，编程思路有了有些改变。首先介绍最简单的使用 TMA 的 producer-consumer 模式。到 FA3 里，已经完全使用 `Q` 维度的 parallel 了，算法的输入可以简化成一个 `Qi` block。
## FA3 without intra-warp overlapping
* 输入：`Qi in [Br, d], K V in [N,d]` in HBM, KV tile size `Br`，流水线深度 `s`
* 初始化流水线所需要的环形 buffer，需要存储 `Ki, Vi` 以及同步的barrier 等数据
* `if is producer`
    * 调用 `setmaxnreg` 设置寄存器数量
    * 发射 `load Qi`
    * `for j in range(Tc)`
        * 等待第 `j mod s` 的环形缓冲区被消费掉（初始化时是被消费的状态）
        * 发射 `load Ki, load Vi` to SHM
        * 等待加载完成，`commit to notify consumer` 加载完成
    * end for
* `else`
    * 调用 `setmaxnreg` 设置寄存器数量
    * on-chip 初始化 `O, l, m`
    * 等待 `Qi` 加载完成
    * `for j in range(Tc)`
        * 等待 `Kj` 加载完成
        * on-chip 调用 wgmma 计算 `Sij = Qi @ Kj^T`, SS-GEMM (shm-shm)， `commit and wait`
        * `m_old = rowmax(Sij)`
        * `m = max(m, m_old)`
        * `Pij = exp(Sij-m)`
        * `l = exp(m_old-m)*l + rowsum(Pij)`
        * 等待 `Vj` 加载完成
        * `O = O*exp(m_old-m) Pij@V` RS-GEMM (Register-Shm), `commit and wait`
        * 释放 buffer 给 consumer
    * `end for range Tc`
    * `O_i HBM <- O/l`
* `end`

这里对于计算 warpgroup 可以采用ping-pong 的调度方式来实现 warp 之间的 overlap。计算过程可以拆分成几部分，SS-GEMM, Softmax, RS-GEMM。假设我们有两个warpgroup，则可以如下调度,
GEMM1 → GEMM0 → Softmax。两个 warp 每执行一次后，交换。

## FA3 intra—warp overlapping
还有另外一个 overlap 策略，就是warp 内的 overlap。我们可以拆成多个 iter 来看这个 overlap 过程，即我们希望 `iter_i` 的 RS-GEMM 可以和 `iter_i+1` 的 Softmax 做 overlap。

对于计算warpgroup，伪代码如下
* 调用 `setmaxnreg` 设置寄存器数量
* on-chip 初始化 `O, l, m`
* 等待 `Qi`，`K0` 加载完成
* 计算 `S_cur = Qi @ K0`, `commit and wait`
* 释放 `K0` 的内存
* 基于 `S_cur` 计算 `m l P_cur`
* `for j in range(1,Tc)`
    * 等待 `Kj` load
    * 计算 `S_next = Qi @ Kj`, `commit but NOT wait`
    * 等待 `Vj-1` load
    * 计算 `O = O + P_cur@Vj-1`, `commit but NOT wait`
    * 等待 `S_next` 计算
    * 基于 `S_next` 计算 `m l P_next`
    * 等待 `P_cur@Vj-1` 计算，rescale `O`
    * 释放相应的 KV buffer
    * `S_cur <- S_next`
* `end for`
* Wait for V𝑇𝑐−1 to be loaded in shared memory.
* Compute O𝑖 = O𝑖 + P˜，Commit and wait.
* Epilogue: Rescale O𝑖 based on 𝑚𝑖， Write O𝑖 and 𝐿𝑖 to HBM