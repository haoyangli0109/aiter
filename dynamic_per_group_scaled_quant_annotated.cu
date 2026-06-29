// =============================================================================
//  aiter dynamic_per_group_scaled_quant  逐行中文注释学习版（仅 MXFP4 视角）
//  原文件：/home/haoyanli/lhy_atom/aiter/csrc/kernels/quant_kernels.cu
//  ★这是【未修改的原版】★：MXFP4 的 e8m0 scale 用项目默认 RoundUp(ceil_pow2)，
//   而不是我们后来在 /app/aiter-test 副本里改成的 Even。本文件讲解的就是原始行为。
//
//  这是 aiter "通用 per-group 量化" kernel，支持 fp8 / i8 / fp4 多种输出，
//  group_size 可为 32/64/128。本注释只聚焦【MXFP4(输出 fp4, e8m0 scale)】这条路。
//
//  和另两个 kernel 的关系（详见文件末尾"对比总结"）：
//    - aiter PR#2976 quant_mxfp4.cu  ：FP4 专用、单 kernel 端到端、含打包+shuffle、scale 用 Even
//    - Quark funcs.cuh               ：通用元素级"假量化"(RNE)，只做元素一步
//    - 本 kernel(原版)               ：通用多 dtype per-group 量化，fp4 是其中一条分支，
//                                      靠 opus/硬件 cvt 指令转 fp4，scale 用【RoundUp】
//
//  本文件仅用于阅读学习，不参与编译。
// =============================================================================

#include "aiter_hip_common.h"
#include "aiter_dispatch.h"
#include "aiter_opus_plus.h"   // opus 的 fp32/bf16/fp16 -> fp4 打包转换 + 硬件 cvt 指令
#include "aiter_stream.h"
#include "gemm_dispatch_utils.h"
#include "quant.h"
#include "mx_quant_utils.h"    // fp_f32_to_e8m0_scale<round_mode, dtype>(...) 与 shuffle 下标
#include <hipcub/hipcub.hpp>   // 组内归约 multithread_reduce 用

const int32_t BlockSize           = 256;
const int32_t groupQuantBlockSize = 64;

namespace aiter {

// =============================================================================
//  GPU kernel：每个 group(MXFP4 下 32 个元素) 算 scale + 量化 + 打包写出
//  模板参数（MXFP4 实例化时的取值见后面 host 端）：
//    DTYPE_I  : 输入元素类型(bf16/fp16)
//    DTYPE_O  : 输出类型(MXFP4 时 = opus::fp4_t)
//    thread_data_size=32 : 每线程负责的元素个数
//    group_size=32(MXFP4) : 一个 MX block 的元素数(共享 1 个 e8m0 scale)
//    shuffle_scale : scale 是否按硬件瓦片重排
//    block_size=64 : 线程块大小
//    emit_e8m0_scale : fp8 是否也输出 e8m0(fp4 恒为 true)
// =============================================================================
template <typename DTYPE_I, typename DTYPE_O, int thread_data_size = 32,
          int32_t group_size = 128, bool shuffle_scale = true,
          int32_t block_size = 64, bool emit_e8m0_scale = false>
__global__ void __launch_bounds__(block_size)
dynamic_per_group_scaled_quant_kernel(DTYPE_O* __restrict__ out,        // 输出(fp4 打包/fp8/i8)
                                      float* __restrict__ scale,         // 输出 scale(e8m0 按字节写)
                                      DTYPE_I const* __restrict__ input, // 输入 bf16/fp16
                                      float const* __restrict__ scale_ub,// (本路未用)scale 上界
                                      int64_t ori_rows,
                                      int32_t ori_cols,
                                      int32_t ori_row_stride,
                                      int64_t oob_size,                  // 输出越界保护字节数
                                      int32_t const* __restrict__ num_rows = nullptr,
                                      const int32_t num_cols_factor        = 1)
{
    // fp4/fp8 才允许 e8m0 字节 scale。
    static_assert(!emit_e8m0_scale
                      || std::is_same_v<DTYPE_O, opus::fp4_t>
                      || std::is_same_v<DTYPE_O, opus::fp8_t>,
                  "emit_e8m0_scale is only valid for fp4 / fp8 outputs");

    // fp4 永远用 e8m0 字节 scale；fp8 仅当 emit_e8m0_scale 时用。
    static constexpr bool use_e8m0_scale =
        std::is_same_v<DTYPE_O, opus::fp4_t> || emit_e8m0_scale;

    // 动态行数(MoE 变长场景)：用运行期 num_rows 覆盖。
    if(num_rows != nullptr)
        ori_rows = static_cast<int64_t>(*num_rows) * num_cols_factor;

    // ---- 1) 线程 -> (行 x, 第几个 group y) 的映射 ----
    // num_thread_per_group：一个 group 由几个线程协作（MXFP4: 32/32 = 1，即 1 线程 1 group）。
    static constexpr int num_thread_per_group = group_size / thread_data_size;
    int64_t row_offset = static_cast<int64_t>(blockIdx.x) * block_size;
    int64_t groupId    = (row_offset + threadIdx.x) / num_thread_per_group; // 全局 group 号
    int32_t scaleN     = ori_cols / group_size;                            // 每行 group 数 = cols/32
    // e8m0+shuffle 时，scale 列方向要按 8 对齐(瓦片宽度)。
    int32_t scaleN_pad = (use_e8m0_scale && shuffle_scale)
                             ? (((scaleN + 7) / 8) * 8)
                             : scaleN;
    int64_t x = groupId / scaleN_pad;                 // 行号
    int32_t y = static_cast<int32_t>(groupId % scaleN_pad); // 行内第几个 group

    // 越界线程退出(含 padding 多出的 y)。
    if constexpr(use_e8m0_scale) { if(x >= ori_rows || y >= scaleN) return; }
    else                        { if(x >= ori_rows) return; }

    // ---- 2) 定位本 group 在输入里的起点，并批量读入 ----
    row_offset  = x * ori_row_stride + y * group_size;          // 输入元素偏移
    using vec_i = opus::vector_t<DTYPE_I, thread_data_size>;     // 一次读 32 个元素的向量类型
    // fp4 输出按 2 个 1 字节打包，所以输出向量长度减半。
    static constexpr int32_t vec_size_o =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? thread_data_size / 2 : thread_data_size;

    // 连续 fp32-scale 路(fp8/i8)用的除子 1/DTYPE_MAX；e8m0 路不走它。
    const float inverted_DTYPE_MAX = (1. / static_cast<float>(opus::finfo<DTYPE_O>::max()));

    auto const* input_vecs = reinterpret_cast<vec_i const*>(input + row_offset);
    vec_i thread_data = input_vecs[threadIdx.x % num_thread_per_group]; // 读入本线程的 32 个元素

    // ---- 3) 求组内绝对值最大 absMax ----
    float absMax = 1e-10f;                                       // 防全 0(避免除/对数异常)
    for(size_t j = 0; j < thread_data_size; j++)
        absMax = max(absMax, abs(static_cast<float>(thread_data[j])));
    // 若一个 group 跨多线程(group_size>32)，跨线程归约取整组最大；MXFP4(1线程1组)时是恒等。
    absMax = multithread_reduce(absMax, hipcub::Max(), num_thread_per_group);

    // ---- 4) 由 absMax 求 e8m0 dequant scale ----
    // ★原版行为★：MXFP4 这里用【项目默认 round mode = kDefaultMxScaleRoundMode】，
    //   当前即 RoundUp(ceil_pow2，NV/DSv4 的 RCEIL 语义)，而【不是】Quark/OCP 的 Even。
    //   fp4 与 fp8 共用同一个 round mode，仅 dtype 常量不同。
    //   helper 返回的是 dequant scale(2 的幂)，(>>23)&0xFF 即可抽出 e8m0 字节。
    float inverted_scale;   // 注意：fp4 路径里这其实是"dequant scale"(2 的幂)，不是它的倒数
    if constexpr (use_e8m0_scale)
    {
        // MX dtype 常量：fp4 -> FP4_E2M1；fp8 -> 视架构 FNUZ/标准 E4M3。
        constexpr aiter::MxDtype kMxDtype =
            std::is_same_v<DTYPE_O, opus::fp4_t>
                ? aiter::MxDtype::FP4_E2M1
#if defined(__gfx942__)
                : aiter::MxDtype::FP8_E4M3_FNUZ;
#else
                : aiter::MxDtype::FP8_E4M3;
#endif
        // ★这里是原版与 Quark 的关键差异点★：
        //   用 kDefaultMxScaleRoundMode(RoundUp)，所以 block scale 会比 Quark 的 Even 偏大一档
        //   (当 amax 落在两个 2 的幂中间时，RoundUp 向上取，Even 取最近偶)。
        //   —— 这正是后来我们在副本里把它改成 MxScaleRoundMode::Even 的原因。
        inverted_scale =
            aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, kMxDtype>(absMax);
    }
    else
    {
        // 连续 fp32-scale 路(fp8/i8)：scale = absMax / DTYPE_MAX。
        inverted_scale = absMax * inverted_DTYPE_MAX;
    }

    // ---- 5) 计算本线程输出写出偏移(fp4 打包后字节数减半) ----
    row_offset = std::is_same_v<DTYPE_O, opus::fp4_t>
                     ? groupId * group_size / 2 + (threadIdx.x % num_thread_per_group) * vec_size_o
                     : groupId * group_size     + (threadIdx.x % num_thread_per_group) * vec_size_o;

    // ---- 6) 每组的"组长线程"负责写出 1 个 scale ----
    if(threadIdx.x % num_thread_per_group == 0)
    {
        if constexpr(use_e8m0_scale)
        {
            auto* tmp        = reinterpret_cast<uint8_t*>(scale);
            // 从 dequant scale 的 fp32 位里抽取 8 位 biased 指数 = e8m0 字节。
            uint8_t exponent = (__builtin_bit_cast(uint32_t, inverted_scale) >> 23) & 0b11111111;
            if constexpr(shuffle_scale)   // 需要瓦片重排时，换算重排后的下标
                groupId = aiter::mx_scale_shuffle_idx(scaleN_pad, static_cast<int>(x), y);
            tmp[groupId] = exponent;       // 写 1 字节 e8m0
        }
        else
        {
            if constexpr(shuffle_scale)
                groupId = y * ori_rows + x;     // fp32-scale 的(转置)重排
            scale[groupId] = inverted_scale;    // 写 fp32 scale
        }
    }

    // ---- 7) 把 dequant scale 调整成"存储路径需要的乘子" ----
    // fp4：硬件 cvt_scalef32_pk_fp4 指令直接吃 dequant scale(2 的幂)，所以保持原值；
    // fp8/i8：软件 input * inv_scale，需要倒数 1/scale。
    // (注释原文还记录了一个历史 bug：早期该判断错放在 use_e8m0_scale 上，导致
    //  fp8+e8m0 路径漏取倒数，fp8 字节差约 2 倍。)
    inverted_scale =
        std::is_same_v<DTYPE_O, opus::fp4_t> ? inverted_scale : 1.0f / inverted_scale;

    // ---- 8) 量化 + 打包 + 写出 ----
    // fp4 输出按 uint8 容器存(每字节 2 个 fp4)。
    using DTYPE_STORE = std::conditional_t<std::is_same_v<DTYPE_O, opus::fp4_t>, uint8_t, DTYPE_O>;
    auto* out_ptr = reinterpret_cast<DTYPE_STORE*>(out);
    auto buffer_o = opus::make_gmem<DTYPE_STORE>(out_ptr, oob_size);   // 带越界保护的全局写缓冲
    // store_vector 内部：先按总字节挑 chunk(16/8/4)，再对 fp4 调 opus 的
    //   scaled_cast -> fp32/bf16/fp16_to_fp4_packed，把元素按 scale 缩放并转 fp4。
    //   ★注意★ opus 的 fp4 转换【只有硬件版本】：
    //     gfx950(及 gfx1250 部分)：__builtin_amdgcn_cvt_scalef32_pk_fp4_*(精确 RNE)
    //     gfx942 及其它架构        ：是【空 stub，返回全 0】(opus.hpp #else 分支)
    //   => 因此本 kernel 的 fp4 路径实质上只在 gfx950 上可用；gfx942 没有软件回退
    //      (这点与 PR#2976 不同——PR#2976 专门为 gfx942 写了软件 even_round_e2m1)。
    // 元素量化方向：fp4 用 scale(2的幂)经硬件指令缩放并 round 到 E2M1 网格(RNE)，再 2 个打 1 字节。
    store_vector<DTYPE_STORE, DTYPE_I, thread_data_size, RT, false, WARP_SIZE, 1, DTYPE_O>(
        buffer_o, thread_data, row_offset, inverted_scale);
}

// =============================================================================
//  Host 端入口：校验 -> 选 e8m0/fp32 路 -> 按 group_size/dtype/shuffle 分发模板 -> launch
//  （只保留与 MXFP4 相关的主干逻辑注释）
// =============================================================================
void dynamic_per_group_scaled_quant(aiter_tensor_t& out,         // 输出 [..., d]
                                    const aiter_tensor_t& input, // 输入 [..., d]
                                    aiter_tensor_t& scales,      // 输出 scale
                                    int group_size,              // 32/64/128
                                    bool shuffle_scale,          // scale 是否瓦片重排
                                    std::optional<aiter_tensor_t> num_rows, // 变长行数(MoE)
                                    int num_rows_factor)
{
    AITER_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
                __func__, " only support group_size [32, 64 , 128]");
    AITER_CHECK(out.is_contiguous());

    int const cols       = input.size(-1);
    int const rows       = input.numel() / cols;       // 把高维拍平成 (rows, cols)
    int const row_stride = input.stride(-2);
    int32_t* num_rows_ptr = num_rows.has_value()
                                ? reinterpret_cast<int32_t*>(num_rows->data_ptr()) : nullptr;

    AITER_CHECK(cols % group_size == 0, __func__, " cols is not divisible by group_size");

    // 由 scales.dtype 决定走 e8m0(字节) 还是 fp32(连续) scale。
    // (注意 fp4 输出在 kernel 内恒为 e8m0，与此处无关；这里主要影响 fp8。)
    const bool use_e8m0_scale =
        scales.dtype() == AITER_DTYPE_fp8_e8m0 || scales.dtype() == AITER_DTYPE_u8;
    AITER_CHECK(use_e8m0_scale || scales.dtype() == AITER_DTYPE_fp32,
                __func__, " expects scales.dtype in {fp8_e8m0, u8, fp32}, got ",
                AiterDtype_to_str(scales.dtype()));

    HipDeviceGuard device_guard(input.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // DISPATCH_GROUP_SIZE：把运行期 group_size 变成编译期常量 _GS。
    DISPATCH_GROUP_SIZE(group_size,
        static constexpr int thread_data_size     = 32;            // 每线程 32 元素
        static constexpr int num_thread_per_group = _GS / thread_data_size; // MXFP4(_GS=32)->1
        static constexpr int32_t dynGroupQuantBlockSize = 64;      // 块 64 线程
        const int num_group_per_tg = dynGroupQuantBlockSize / num_thread_per_group;

        int scaleN = cols / _GS;                                    // 每行 group 数
        dim3 const block(dynGroupQuantBlockSize);

        // launch：根据 (输出类型, 是否 shuffle, 是否 e8m0) 实例化对应模板并发射。
        auto launch = [&](auto out_type_tag, auto shuffle_tag, auto e8m0_tag) {
            using out_t = decltype(out_type_tag);
            constexpr bool ss = decltype(shuffle_tag)::value;
            constexpr bool ee = decltype(e8m0_tag)::value;
            // e8m0+shuffle 时 scale 槽位数按 8 对齐；否则正好 rows*scaleN。
            int num_group;
            if constexpr(ee) num_group = ss ? rows * ((scaleN + 7) / 8 * 8) : rows * scaleN;
            else             num_group = rows * scaleN;
            // 输出越界保护(按 4 字节对齐凑整)。
            static constexpr int32_t ooba = 4 / sizeof(out_t);
            const int64_t oob_elems =
                (static_cast<int64_t>(rows) * cols + ooba - 1) / ooba * ooba;
            const int64_t oob_size = oob_elems * static_cast<int64_t>(sizeof(out_t));
            dim3 const grid((num_group + num_group_per_tg - 1) / num_group_per_tg);
            // 再按输入 dtype(bf16/fp16) 分发，最终实例化 + launch kernel。
            AITER_DISPATCH_FLOATING16_TYPES_rmTorch(
                input.dtype(), "dynamic_per_group_scaled_quant_kernel", [&] {
                    using input_dtype = typename aiter::hip2opus<scalar_t>::type;
                    aiter::dynamic_per_group_scaled_quant_kernel<
                        input_dtype, out_t, thread_data_size, _GS, ss, dynGroupQuantBlockSize, ee>
                        <<<grid, block, 0, stream>>>(
                        reinterpret_cast<out_t*>(out.data_ptr()),
                        reinterpret_cast<float*>(scales.data_ptr()),
                        reinterpret_cast<input_dtype*>(input.data_ptr()),
                        nullptr, rows, cols, row_stride, oob_size,
                        num_rows_ptr, num_rows_factor);
                });
        };

        // 按输出 dtype 选分支：fp8 / i8 / fp4。
        auto do_launch = [&](auto shuffle_tag, auto e8m0_tag) {
            constexpr bool ee = decltype(e8m0_tag)::value;
            if(out.dtype() == AITER_DTYPE_fp8)        launch(opus::fp8_t{}, shuffle_tag, e8m0_tag);
            else if(out.dtype() == AITER_DTYPE_i8) {
                AITER_CHECK(!ee, __func__, " i8 output does not support e8m0 scale");
                launch(opus::i8_t{}, shuffle_tag, std::false_type{});
            }
#if defined(__Float4_e2m1fn_x2)
            // ★MXFP4 分支★：fp4 输出，e8m0 标签强制 true(fp4 恒用 e8m0 scale)。
            else if(out.dtype() == AITER_DTYPE_fp4x2 || out.dtype() == AITER_DTYPE_u8)
                launch(opus::fp4_t{}, shuffle_tag, std::true_type{});
#endif
            else AITER_CHECK(false, __func__, " not support output type: ",
                             AiterDtype_to_str(out.dtype()));
        };

        // 组合 shuffle / e8m0 两个编译期开关。
        auto with_e8m0 = [&](auto shuffle_tag) {
            if(use_e8m0_scale) do_launch(shuffle_tag, std::true_type{});
            else               do_launch(shuffle_tag, std::false_type{});
        };
        if(shuffle_scale) with_e8m0(std::true_type{});
        else              with_e8m0(std::false_type{});
    )
}

} // namespace aiter

// =============================================================================
//  ★对比总结：dynamic_per_group_scaled_quant(原版)  vs  aiter PR#2976  vs  Quark★
// =============================================================================
//
// 一、三者定位
//   - 本 kernel(dynamic_per_group_scaled_quant，原版)
//       通用 per-group "在线量化" kernel：一套代码支持 fp8 / i8 / fp4，
//       group_size 32/64/128。fp4 只是其中一条编译期分支。
//   - aiter PR#2976 (quant_mxfp4.cu)
//       FP4 专用、单 kernel 端到端：算 scale + 量化 + 打包 + 多种 shuffle，
//       group_size 固定 32，为 MXFP4 高度特化、吞吐更高。
//   - Quark funcs.cuh (fake_quantize_element)
//       通用"元素级假量化"(fp4/6/8 通用)，输出 fp32、不打包、不算 scale、不 shuffle，
//       是数学参考 / 离线对齐实现；scale 与打包由上层 Python(even_round + Pack_fp4) 串。
//
// 二、scale(e8m0) 的舍入  ←★这是本(原版)kernel 与另两者最大的不同★
//   - 本 kernel(原版)：fp_f32_to_e8m0_scale<kDefaultMxScaleRoundMode=RoundUp, FP4>
//       → RoundUp(ceil_pow2)：amax 落在两个 2 的幂之间时【向上取一档】。
//   - PR#2976 ：位运算 even-round → Even。
//   - Quark   ：even_round() 位运算 → Even。
//   => 原版 kernel 的 block scale 会比 Quark/PR#2976 偏大一档(在临界值上)，
//      这就是它和 Quark 不逐字节一致的主因；后来我们把副本改成 Even 才对齐。
//
// 三、元素 -> FP4 的舍入
//   - 本 kernel：store_vector -> opus scaled_cast -> *_to_fp4_packed
//       gfx950 用硬件 cvt 指令(精确 RNE)；gfx942 用 opus 软件实现。
//   - PR#2976 ：gfx950 同样硬件 cvt(RNE)；gfx942 软件 even_round_e2m1(阈值法, half-away)。
//   - Quark   ：位级 shift_and_round(精确 RNE, tie 取偶)。
//   => 在 gfx950 上元素舍入都是 RNE，趋于一致。
//
// 四、零的符号(±0)
//   - 本 kernel / PR#2976：走硬件 cvt，负数 flush 到 0 时保留符号 -> 可能产生 -0(0x8)。
//   - Quark hw_emulation：if(mantissa==0) return 0.0 -> 恒 +0。
//
// 五、性能 / 完整性
//   - 本 kernel：融合(scale+量化+打包+可选 shuffle)、向量化、硬件 cvt，性能高；
//                通用多 dtype，但 fp4 没有 PR#2976 那么多专用 weight-shuffle 布局。
//   - PR#2976 ：FP4 最专、shuffle 布局最全，直接喂 GEMM/MoE。
//   - Quark   ：最慢(逐元素 fp32 + 多次 launch + 大中间张量)，但数学最正确、最通用、最易读。
//
// 六、一句话
//   原版 dynamic_per_group 与 Quark 的核心差异在【scale 用 RoundUp 而非 Even】(临界值差一档)，
//   外加 gfx942 元素舍入与 ±0 符号两处小差异。把 scale 改成 Even(我们在 /app/aiter-test 做的)
//   后，scale 即与 Quark 逐字节一致，量化值仅余 ±0 的等价差异。
// =============================================================================
