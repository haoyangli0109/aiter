#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <type_traits>
#include <stdint.h>
#include <math.h>
#include <iostream>
#endif

#include "hip_float8_impl.h"

struct alignas(1) hip_fp8  // 定义一个结构体 hip_fp8，要求按1字节对齐
{
  struct from_bits_t {};  // 声明一个空结构体 from_bits_t，作为特殊构造函数的标志

  HIP_FP8_HOST_DEVICE static constexpr from_bits_t from_bits()  // 静态函数返回一个 from_bits_t 对象，用于构造器标志
  {
    return from_bits_t();  // 返回空对象
  }

  uint8_t data;  // 实际存储的 FP8 原始数据（1字节）

  hip_fp8() = default;  // 默认构造函数

  HIP_FP8_HOST_DEVICE constexpr hip_fp8(const hip_fp8 &) = default;  // 拷贝构造函数

  HIP_FP8_HOST_DEVICE constexpr hip_fp8(uint8_t v) = delete;  // 禁止通过 uint8_t 直接构造（防止误用）

  explicit HIP_FP8_HOST_DEVICE constexpr hip_fp8(uint8_t v, from_bits_t)
      : data(v) {}  // 允许通过 uint8_t 和 from_bits_t 标志构造，用于从原始比特初始化

#ifdef __HIP__MI300__
  explicit HIP_FP8_DEVICE hip_fp8(float v)
      : data(hip_fp8_impl::to_fp8_from_fp32(v)) {}  // MI300设备上：从 float 构造 FP8，调用设备函数转换

  explicit HIP_FP8_DEVICE hip_fp8(_Float16 v)
      : hip_fp8(static_cast<float>(v)) {}  // 从 _Float16 构造，先转换为 float

  explicit HIP_FP8_HOST
#else
  explicit HIP_FP8_HOST_DEVICE
#endif
  hip_fp8(float v)
  {
    data = hip_fp8_impl::to_float8<4, 3, float, true, true>(v);  // 非 MI300 上：用模拟方式转换 float 到 FP8
  }

  explicit HIP_FP8_HOST_DEVICE hip_fp8(double v)
      : hip_fp8(static_cast<float>(v)) {}  // 从 double 构造，先转成 float，再转成 fp8

#ifdef __HIP__MI300__
  explicit inline HIP_FP8_DEVICE operator float() const
  {
    float fval;
    uint32_t i32val = static_cast<uint32_t>(data);  // 将 data 转为 32 位整数以用于汇编

    asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0"
                 : "=v"(fval)
                 : "v"(i32val));  // 使用汇编指令将 FP8 转 float

    return fval;
  }

  explicit inline HIP_FP8_HOST operator float() const
#else
  explicit inline HIP_FP8_HOST_DEVICE operator float() const
#endif
  {
    return hip_fp8_impl::from_float8<4, 3, float, true>(data);  // 非 MI300 上：使用软件模拟方式转成 float
  }
};


namespace std
{
  inline hip_fp8 sin(hip_fp8 a) { return hip_fp8(sinf(float(a))); }
  inline hip_fp8 cos(hip_fp8 a) { return hip_fp8(cosf(float(a))); }
  HIP_FP8_HOST_DEVICE constexpr hip_fp8 real(const hip_fp8 &a) { return a; }
} // namespace std

// Special operator overloading
inline std::ostream &operator<<(std::ostream &os, const hip_fp8 &f8)
{
  return os << float(f8);
}

// all + operator overloading with mixed types
// mixed types, always converts to f32, does computation in f32, and returns
// float
inline HIP_FP8_HOST_DEVICE float operator+(const float fa, hip_fp8 b)
{
  return (fa + float(b));
}

inline HIP_FP8_HOST_DEVICE float operator+(hip_fp8 a, const float fb)
{
  return (float(a) + fb);
}

inline HIP_FP8_HOST_DEVICE hip_fp8 operator+(hip_fp8 a, hip_fp8 b)
{
  return hip_fp8(float(a) + float(b));
}

inline HIP_FP8_HOST_DEVICE hip_fp8 &operator+=(hip_fp8 &a, hip_fp8 b)
{
  return a = hip_fp8(float(a) + float(b));
}

// overloading multiplication, always returns float,
inline HIP_FP8_HOST_DEVICE float operator*(hip_fp8 a, hip_fp8 b)
{
  return float(a) * float(b);
}

inline HIP_FP8_HOST_DEVICE float operator*(float a, hip_fp8 b)
{
  return (a * float(b));
}

inline HIP_FP8_HOST_DEVICE float operator*(hip_fp8 a, float b)
{
  return (float(a) * b);
}

inline HIP_FP8_HOST_DEVICE float operator*(int32_t a, hip_fp8 b)
{
  return ((float)a * float(b));
}

inline HIP_FP8_HOST_DEVICE float operator*(double a, hip_fp8 b)
{
  return ((float)a * float(b));
}

// overloading for compare
inline HIP_FP8_HOST_DEVICE bool operator==(hip_fp8 a, hip_fp8 b)
{
  return (a.data == b.data);
}
inline HIP_FP8_HOST_DEVICE bool operator!=(hip_fp8 a, hip_fp8 b)
{
  return (a.data != b.data);
}

inline HIP_FP8_HOST_DEVICE bool operator>=(hip_fp8 a, hip_fp8 b)
{
  return static_cast<float>(a) >= static_cast<float>(b);
}
inline HIP_FP8_HOST_DEVICE bool operator>(hip_fp8 a, hip_fp8 b)
{
  return static_cast<float>(a) > static_cast<float>(b);
}
