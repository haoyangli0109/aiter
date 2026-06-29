// =============================================================================
//  PR ROCm/aiter#2976 —— csrc/kernels/quant_mxfp4.cu  逐行中文注释学习版
//  原始内容：275 行全新文件（quantize 专用 MXFP4 kernel，round_mode=Even）
//  本文件仅用于"阅读学习"，注释为中文讲解，不参与实际编译。
// =============================================================================

// SPDX-License-Identifier: MIT                       // 开源许可证：MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.  // 版权声明

#include "aiter_hip_common.h"   // aiter 通用 HIP 宏/类型：AITER_CHECK、vector_t、u8x16_t 等
#include "aiter_dispatch.h"     // 类型分发宏：AITER_DISPATCH_FLOATING16_TYPES_rmTorch（fp16/bf16 -> scalar_t）
#include "aiter_opus_plus.h"    // 底层设备函数/内建指令封装（fp4/fp8 转换、cvt 指令等）
#include "aiter_stream.h"       // 获取当前 HIP stream：aiter::getCurrentHIPStream()
#include "quant.h"              // 本算子对外声明：void aiter::quant_mxfp4(...)

namespace aiter {               // 所有符号放进 aiter 命名空间，避免与外部冲突

// -----------------------------------------------------------------------------
// 舍入模式枚举
// -----------------------------------------------------------------------------
// Even：e8m0 块缩放(scale)用"四舍六入五成双(round-half-to-even)"把组内最大值
//       round 到最近的 2 的幂。
//   - gfx950：用硬件内建指令（精确 RNE，round-to-nearest-even）。
//   - gfx942：用软件回退实现（round-half-away，五入到远离 0 的方向）。
// 这里只定义了一个取值 Even=0，说明该 kernel 目前只支持 Even 一种 scale 舍入。
enum class MxFp4RoundMode : int { Even = 0 };

// -----------------------------------------------------------------------------
// Even 舍入用到的位运算常量（核心：直接在 fp32 的位模式上做"偶数舍入到 2 的幂"）
// -----------------------------------------------------------------------------
// 0x7F800000：fp32 的 [符号位 + 8 位指数] 掩码（即清掉 23 位尾数，只保留 sign+exp）。
#define EVEN_ROUND_FP32_SIGN_EXP_MASK 0x7F800000u
// 0x00200000：在尾数最高位(第 22 位)上加 1，相当于给指数部分做"进位舍入"。
//   尾数 23 位，0x00200000 = 1<<21 ……实际上是尾数的 1/4 位置；配合后面截断尾数，
//   实现把 max 舍入到最近的 2 的幂（向偶数指数靠拢）。
#define EVEN_ROUND_VAL_TO_ADD 0x00200000u
// FP4(E2M1) 的最大规格化指数 emax = 2（对应最大可表示值 6.0 = 1.5 * 2^2）。
#define EVEN_ROUND_FP4_EMAX 2

// -----------------------------------------------------------------------------
// 编译期常量
// -----------------------------------------------------------------------------
static constexpr int kGroupSize      = 32;             // MX 块大小：每 32 个元素共享一个 e8m0 scale
static constexpr int kPackedPerGroup = kGroupSize / 2; // 每组打包后字节数：32 个 fp4 -> 16 字节(每字节 2 个 fp4)
static constexpr int kBlockThreads   = 256;            // 每个线程块 256 线程（一个线程处理一个 group）

// 一次性读取 8 个 uint16（=16 字节）的向量类型，用于把输入按 128-bit 对齐批量加载。
using packed_u16x8_t = vector_t<uint16_t, 8>;

// =============================================================================
//  设备端(__device__)辅助函数：fp4 转换
// =============================================================================
#if defined(__gfx950__)
// ---- gfx950(MI350 等)：有原生 scalef32 -> packed fp4 的硬件转换指令 ----
template <typename ftype, int sel>   // ftype: 输入元素类型(bf16/fp16)；sel: 4 个 pack 中的第几个槽位(0..3)
__device__ __forceinline__ uint32_t cvt_fp4_pk(uint32_t src, uint32_t pair, float scale) {
  // 把"一对(2 个)"bf16/fp16，按 scale 缩放后转成 2 个 fp4，写入 src 的第 sel 个 nibble 位置。
  if constexpr (std::is_same_v<ftype, hip_bfloat16>)
    return __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(   // bf16x2 -> fp4x2 的硬件指令
      src, __builtin_bit_cast(bf16x2_t, pair), scale, sel);
  else
    return __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(    // fp16x2 -> fp4x2 的硬件指令
      src, __builtin_bit_cast(fp16x2_t, pair), scale, sel);
}
#else
// ---- gfx942(MI300 等)：无原生指令，用软件实现把 1 个 float 量化成 1 个 E2M1(fp4) nibble ----
__device__ __forceinline__ uint8_t even_round_e2m1(float val) {
  float a = fabsf(val);          // 取绝对值，符号最后单独处理
  uint8_t mag;                   // fp4 的"幅值"部分(3 位：1 exp 高位 + ……其实是 e2m1 的低 3 位)
  // 下面这串阈值就是 E2M1 可表示值 {0,0.5,1,1.5,2,3,4,6} 的"中点判决边界"，
  // 即把 a 四舍五入到最近的可表示幅值；边界取在两个相邻可表示值的中点：
  //   6 与 4 的中点=5；4 与 3 的中点=3.5；3 与 2 的中点=2.5；2 与 1.5 的中点=1.75；
  //   1.5 与 1 的中点=1.25；1 与 0.5 的中点=0.75；0.5 与 0 的中点=0.25。
  if      (a >= 5.0f)  mag = 7;  // -> 6.0
  else if (a >= 3.5f)  mag = 6;  // -> 4.0
  else if (a >= 2.5f)  mag = 5;  // -> 3.0
  else if (a >= 1.75f) mag = 4;  // -> 2.0
  else if (a >= 1.25f) mag = 3;  // -> 1.5
  else if (a >= 0.75f) mag = 2;  // -> 1.0
  else if (a >= 0.25f) mag = 1;  // -> 0.5
  else                 mag = 0;  // -> 0.0
  uint8_t sign_bit = (val < 0.0f) ? 8u : 0u;  // fp4 的最高位是符号位(0x8)；负数置 1
  return sign_bit | mag;                       // 拼成最终 4-bit 编码(sign|mag)
}
#endif

// =============================================================================
//  设备端辅助函数：scale(e8m0) 的"洗牌(shuffle)"地址映射
//  当 scale 要按特定硬件布局存放(供后续 GEMM/MoE 直接消费)时，计算其落点下标。
// =============================================================================
// e8m0_shuffle 模式下，scale 的目标下标计算（按 32 行 x 8 列的瓦片重排）。
__device__ __forceinline__ int fp4_scale_shuffle_id(int scaleN_pad, int x, int y) {
  return (x / 32 * scaleN_pad) * 32 +     // 行方向以 32 为一组的大块偏移
         (y / 8) * 256 +                  // 列方向以 8 为一组的瓦片偏移
         (y % 4) * 64 +                   // 组内列细分
         (x % 16) * 4 +                   // 组内行细分
         (y % 8) / 4 * 2 +                // 8 列内的上/下半区
         (x % 32) / 16;                   // 32 行内的上/下半区
}

// a16w4_shuffle 模式下(activation16-weight4 GEMM 专用布局)，scale 的目标下标计算。
__device__ __forceinline__ int a16w4_shuffle_scale_id(
  int scaleN, int ori_rows, int x, int y, bool gate_up
) {
  int N1_idx, N_Pack_idx, N_Lane_idx;       // 行方向(N)拆成 3 级索引
  if (gate_up) {                            // gate_up：gate/up 两个权重拼一起的特殊布局
    int half_rows = ori_rows / 2;           // 上半为 gate、下半为 up
    N_Pack_idx = x / half_rows;             // 属于 gate(0) 还是 up(1)
    int rem    = x % half_rows;             // 在各自半区内的行号
    N1_idx     = rem / 16;                  // 每 16 行一个 lane 组
    N_Lane_idx = rem % 16;                  // 组内 lane 号
  } else {
    N1_idx     = x / 32;                    // 每 32 行一组
    N_Pack_idx = (x % 32) / 16;             // 组内上/下 16 行
    N_Lane_idx = x % 16;                    // 16 行内 lane 号
  }
  int K1_idx     = y / 8;                   // 列方向(K)每 8 列一组
  int K_Pack_idx = (y % 8) / 4;             // 组内上/下 4 列
  int K_Lane_idx = y % 4;                   // 4 列内 lane 号
  int k1_size    = scaleN / 8;              // 列方向一共有多少个 8-列组
  return N1_idx * (k1_size * 256) +         // 按上面拆出的多级索引拼出最终线性地址
         K1_idx * 256 + K_Lane_idx * 64 + N_Lane_idx * 4 +
         K_Pack_idx * 2 + N_Pack_idx;
}

// =============================================================================
//  主 kernel：每个线程负责一个 group(32 个元素)，算 scale + 量化 + 打包 + 写出
// =============================================================================
template <typename float_type,        // 输入元素类型(bf16/fp16)
          MxFp4RoundMode rmode,       // scale 舍入模式(此版本恒为 Even)
          bool e8m0_shuffle,          // scale 是否按 e8m0 瓦片重排
          bool a16w4_shuffle,         // scale/权重是否按 a16w4 GEMM 布局重排
          bool shuffle_weight>        // 打包后的权重是否也要重排
__global__ __launch_bounds__(kBlockThreads)   // 限定每块最多 256 线程，利于寄存器分配
void quant_mxfp4_kernel(
  const float_type* __restrict__ inp,   // 输入：[ori_rows, ori_cols] 的 bf16/fp16，行主序
  uint8_t*          __restrict__ out_packed, // 输出：打包后的 fp4(每字节 2 个 fp4)
  float*            __restrict__ out_scale,  // 输出：e8m0 scale(此处按 float* 传入，但实际只写 1 字节)
  int64_t ori_rows, int32_t ori_cols,        // 原始形状
  int32_t scaleN, int32_t scaleN_pad,        // 每行的 scale 个数 / 对齐 padding 后的个数
  bool gate_up                               // a16w4 gate_up 布局开关(运行期值)
) {
  // ---- 1) 计算本线程负责哪个 (行 x, 列组 y) ----
  int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程号
  int64_t x   = gid / scaleN_pad;   // 行号(用 padding 后的列组数来除，保证布局对齐)
  int32_t y   = gid % scaleN_pad;   // 列组号(0..scaleN_pad-1)

  if (x >= ori_rows || y >= scaleN) return;  // 越界线程(含 padding 多出来的 y)直接退出

  // ---- 2) 把这一组 32 个元素从全局内存批量读入寄存器 ----
  // 起始地址 = inp + 行偏移 + 列组偏移；一个 group=32 元素=64 字节(bf16)=4 个 16字节向量。
  const packed_u16x8_t* vp = reinterpret_cast<const packed_u16x8_t*>(
    inp + x * ori_cols + y * kGroupSize
  );
  packed_u16x8_t chunks[4];          // 4 个 128-bit 向量 = 32 个 16-bit 元素
  #pragma unroll
  for (int i = 0; i < 4; ++i) chunks[i] = vp[i];   // 向量化加载(每次 16 字节)

  // 把按 16-bit 读进来的原始位，重新解释成 float_type(bf16/fp16) 的数组视图。
  const float_type* elems = reinterpret_cast<const float_type*>(chunks);

  // ---- 3) 求组内绝对值最大 group_max ----
  float group_max = 0.f;
#if defined(__gfx950__)
  // gfx950 后面用硬件指令直接吃原始 pair，不需要展开成 float 数组，只需 group_max。
  #pragma unroll
  for (int i = 0; i < kGroupSize; ++i)
    group_max = fmaxf(group_max, fabsf(static_cast<float>(elems[i])));
#else
  // gfx942 软件路径：顺便把每个元素转成 float 存到 vals[]，后面量化要用。
  float vals[kGroupSize];
  #pragma unroll
  for (int i = 0; i < kGroupSize; ++i) {
    vals[i]   = static_cast<float>(elems[i]);
    group_max = fmaxf(group_max, fabsf(vals[i]));
  }
#endif

  // ---- 4) 由 group_max 计算 e8m0 缩放(dequant_scale) 与其字节编码(biased_exp) ----
  uint32_t max_bits = __float_as_uint(group_max);  // 取 group_max 的 fp32 位模式
  float    dequant_scale;                          // 反量化缩放(2 的幂)：x_real ≈ fp4_val * dequant_scale
  uint8_t  biased_exp;                             // e8m0 的字节(就是 fp32 的 8 位 biased 指数)

  if constexpr (rmode == MxFp4RoundMode::Even) {
    // (a) 在位模式上做"偶数舍入到 2 的幂"：先给尾数加偏置，再截掉尾数只留 sign+exp。
    max_bits = (max_bits + EVEN_ROUND_VAL_TO_ADD) & EVEN_ROUND_FP32_SIGN_EXP_MASK;
    float max_rounded = __uint_as_float(max_bits); // 舍入后的(2 的幂)最大值

    // (b) 由舍入后的 max 推出"未加 bias 的指数"，再减去 FP4 的 emax(=2)。
    //     含义：让 fp4 的最大可表示值(6.0=1.5*2^2)对应 max_rounded。
    float scale_unbiased = floorf(log2f(max_rounded)) - EVEN_ROUND_FP4_EMAX;
    // (c) 钳制到 e8m0 合法范围 [-127,127]，防止溢出/下溢出指数表示范围。
    scale_unbiased = fminf(fmaxf(scale_unbiased, -127.0f), 127.0f);
    // (d) 还原成真正的缩放值 2^scale_unbiased。
    dequant_scale  = exp2f(scale_unbiased);
    // (e) 提取该缩放值的 fp32 biased 指数(8 位)，即 e8m0 要存的字节。
    biased_exp     = (__float_as_uint(dequant_scale) >> 23) & 0xFF;
  }

  // ---- 5) 把 32 个元素量化并打包成 16 字节 ----
  u8x16_t packed;   // 输出：16 字节 = 32 个 fp4

#if defined(__gfx950__)
  // gfx950：用硬件 cvt 指令，一次处理一对(pair)，4 个 pair 拼成一个 uint32，共 4 个 uint32=16 字节。
  const uint32_t* pairs = reinterpret_cast<const uint32_t*>(chunks); // 每个 uint32 装一对 bf16/fp16
  u32x4_t pw;                                                        // 4 个 uint32 输出
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t* p = pairs + j * 4;   // 本 uint32 输出对应输入里的 4 个 pair(=8 个元素)
    uint32_t w = 0;
    // sel=0..3：把 4 个 pair 各自转成 fp4x2，依次塞进 w 的 4 个 nibble 槽位。
    w = cvt_fp4_pk<float_type, 0>(w, p[0], dequant_scale);
    w = cvt_fp4_pk<float_type, 1>(w, p[1], dequant_scale);
    w = cvt_fp4_pk<float_type, 2>(w, p[2], dequant_scale);
    w = cvt_fp4_pk<float_type, 3>(w, p[3], dequant_scale);
    pw[j] = w;
  }
  packed = __builtin_bit_cast(u8x16_t, pw);  // 4 个 uint32 -> 16 字节
#else
  // gfx942 软件路径：用 1/dequant_scale 把元素缩放回 fp4 "数轴"，再 even_round_e2m1 编码。
  float quant_scale = (dequant_scale == 0.0f) ? 0.0f : (1.0f / dequant_scale); // 防除零
  #pragma unroll
  for (int i = 0; i < kPackedPerGroup; ++i) {    // 16 次，每次产出 1 字节(2 个 fp4)
    packed[i] = even_round_e2m1(vals[i * 2]     * quant_scale)         // 低 nibble
              | (even_round_e2m1(vals[i * 2 + 1] * quant_scale) << 4); // 高 nibble
  }
#endif

  // ---- 6) 计算打包权重的写出地址 w_base(支持多种重排布局) ----
  int64_t w_base;
  if constexpr (shuffle_weight) {              // 需要把权重按硬件 GEMM 布局重排
    int K_pk = ori_cols / 2;                   // 打包后每行字节数(K 方向)
    if constexpr (e8m0_shuffle) {              // e8m0 配套的权重重排公式
      w_base = (int64_t)(x >> 4) * 16 * K_pk + (x & 15) * 16
             + (y >> 1) * 512 + (y & 1) * 256;
    } else if constexpr (a16w4_shuffle) {      // a16w4 配套的权重重排公式
      int K0 = y >> 2, KLane = y & 3;          // 列方向拆成 K0(组) + KLane(组内)
      if (gate_up) {                           // gate_up 拼接布局
        int half_rows = (int)ori_rows >> 1;
        int N_Pack    = (int)x / half_rows;    // gate(0)/up(1)
        int rem       = (int)x % half_rows;    // 半区内行号
        int K0_size   = K_pk >> 6;             // K0 组数
        w_base = (int64_t)(rem >> 4) * (2 * K0_size * 1024)
               + (int64_t)N_Pack * (K0_size * 1024)
               + K0 * 1024 + KLane * 256 + (rem & 15) * 16;
      } else {                                 // 普通(非 gate_up)a16w4 布局
        w_base = (int64_t)(x >> 4) * 16 * K_pk + (x & 15) * 16
               + K0 * 1024 + KLane * 256;
      }
    }
  } else {                                     // 不重排：按最朴素的行主序连续布局
    w_base = x * (ori_cols / 2) + y * kPackedPerGroup;
  }
  // 一次性写出 16 字节打包权重。
  *reinterpret_cast<u8x16_t*>(out_packed + w_base) = packed;

  // ---- 7) 计算 scale 的写出下标(同样支持重排)，写入 1 字节 e8m0 ----
  int scale_idx;
  if constexpr (e8m0_shuffle) {
    scale_idx = fp4_scale_shuffle_id(scaleN_pad, (int)x, y);
  } else if constexpr (a16w4_shuffle) {
    scale_idx = a16w4_shuffle_scale_id(scaleN, (int)ori_rows, (int)x, y, gate_up);
  } else {
    scale_idx = (int)(x * scaleN + y);   // 朴素布局：行主序
  }
  // out_scale 虽声明为 float*，但 e8m0 实际只占 1 字节，这里按 uint8_t* 写。
  reinterpret_cast<uint8_t*>(out_scale)[scale_idx] = biased_exp;
}

// =============================================================================
//  启动宏：把模板参数(数据类型 + 3 个 bool 重排开关)实例化并 launch
// =============================================================================
#define MXFP4_LAUNCH(ftype, rmode, ss, a16, sw)                       \
  quant_mxfp4_kernel<ftype, rmode, ss, a16, sw>                       \
    <<<(int)grid_size, kBlockThreads, 0, stream>>>(                   \
      reinterpret_cast<const ftype*>(inp.data_ptr()),                 \
      reinterpret_cast<uint8_t*>(out_packed.data_ptr()),             \
      reinterpret_cast<float*>(out_scale.data_ptr()),                \
      ori_rows, ori_cols, scaleN, scaleN_pad, gate_up)

// 根据运行期的 3 个 bool 开关，选出对应的编译期模板实例(把运行期 bool 变成编译期常量，
// 让上面的 if constexpr 都能在编译期裁剪掉死代码 -> 性能更好)。
#define MXFP4_DISPATCH(ftype, rmode)                                  \
  if (e8m0_shuffle) {                                                 \
    if (shuffle_weight) { MXFP4_LAUNCH(ftype, rmode, true, false, true); } \
    else                { MXFP4_LAUNCH(ftype, rmode, true, false, false); } \
  } else if (a16w4_shuffle) {                                         \
    if (shuffle_weight) { MXFP4_LAUNCH(ftype, rmode, false, true, true); } \
    else                { MXFP4_LAUNCH(ftype, rmode, false, true, false); } \
  } else {                                                            \
    MXFP4_LAUNCH(ftype, rmode, false, false, false);                 \
  }

// =============================================================================
//  Host 端入口：参数校验 -> 计算网格 -> 按 dtype 分发 -> launch kernel
// =============================================================================
void quant_mxfp4(
  const aiter_tensor_t& inp,         // 输入张量(bf16/fp16, 2D, 连续)
  aiter_tensor_t&       out_packed,  // 输出：打包 fp4
  aiter_tensor_t&       out_scale,   // 输出：e8m0 scale
  int  group_size,                   // 组大小(必须=32)
  int  round_mode,                   // 舍入模式(必须=0 Even)
  bool e8m0_shuffle,                 // 是否 e8m0 重排
  bool a16w4_shuffle,                // 是否 a16w4 重排
  bool gate_up,                      // 是否 gate_up 拼接布局
  bool shuffle_weight                // 是否重排打包权重
) {
  // ---- 参数合法性检查 ----
  AITER_CHECK(inp.is_contiguous(),        __func__, " expected input to be contiguous");
  AITER_CHECK(inp.dim() == 2,             __func__, " expected 2D input");
  AITER_CHECK(out_packed.is_contiguous(), __func__, " expected out_packed to be contiguous");
  AITER_CHECK(out_scale.is_contiguous(),  __func__, " expected out_scale to be contiguous");
  AITER_CHECK(group_size == 32,           __func__, " expected group_size=32");        // 只支持 32
  AITER_CHECK(round_mode == 0,            __func__, " only Even round mode (0) is supported"); // 只支持 Even
  AITER_CHECK(!(e8m0_shuffle && a16w4_shuffle),                                          // 两种 scale 重排互斥
              __func__, " e8m0_shuffle and a16w4_shuffle are mutually exclusive");
  AITER_CHECK(!shuffle_weight || e8m0_shuffle || a16w4_shuffle,                          // 权重重排需依附某种 scale 重排
              __func__, " shuffle_weight requires e8m0_shuffle or a16w4_shuffle");

  // ---- 取形状、算 scale 列数 ----
  const int64_t ori_rows = inp.size(0);
  const int32_t ori_cols = inp.size(1);
  AITER_CHECK(ori_cols % group_size == 0, __func__, " cols must be divisible by group_size"); // 列必须整除 32

  const int32_t scaleN     = ori_cols / group_size;                             // 每行 scale 个数
  const int32_t scaleN_pad = e8m0_shuffle ? ((scaleN + 7) / 8) * 8 : scaleN;    // e8m0 重排需 8 对齐

  // ---- 各重排模式的额外形状约束 ----
  if (a16w4_shuffle) {
    AITER_CHECK(ori_rows % 32 == 0, __func__, " a16w4 scale shuffle requires rows % 32 == 0");
    AITER_CHECK(scaleN % 8 == 0,    __func__, " a16w4 scale shuffle requires scaleN % 8 == 0");
  }
  if (shuffle_weight) {
    AITER_CHECK(ori_rows % 16 == 0, __func__, " shuffle_weight requires rows % 16 == 0");
    int K_pk = ori_cols / 2;        // 打包后列字节数
    if (e8m0_shuffle) {
      AITER_CHECK(K_pk % 32 == 0, __func__, " e8m0 weight shuffle requires K_pk % 32 == 0");
    } else {
      AITER_CHECK(K_pk % 64 == 0, __func__, " a16w4 weight shuffle requires K_pk % 64 == 0");
    }
  }

  // ---- 计算 grid 大小：总 group 数 / 每块线程数(向上取整) ----
  const int64_t total_groups = ori_rows * (int64_t)scaleN_pad;            // 总 group 数(含 padding)
  const int64_t grid_size    = (total_groups + kBlockThreads - 1) / kBlockThreads; // 块数
  AITER_CHECK(grid_size <= 2147483647LL, __func__, " grid size exceeds maximum");   // 不超过 int 上限

  // ---- 设备/stream 上下文 ----
  HipDeviceGuard device_guard(inp.device_id);                 // 切到输入所在 GPU(RAII 自动恢复)
  const hipStream_t stream = aiter::getCurrentHIPStream();    // 当前 HIP stream

  // ---- 按输入 dtype(fp16/bf16) 分发，把具体 scalar_t 代入模板并 launch ----
  AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
    inp.dtype(), "quant_mxfp4_kernel", [&] {
      MXFP4_DISPATCH(scalar_t, MxFp4RoundMode::Even);   // round_mode 恒为 Even
    });
}

#undef MXFP4_LAUNCH      // 清理宏，避免污染后续编译单元
#undef MXFP4_DISPATCH

} // namespace aiter
