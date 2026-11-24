/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <qnnpack/q8avgpool.h>
#include <qnnpack/q8conv.h>
#include <qnnpack/q8dwconv.h>
#include <qnnpack/q8gavgpool.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/u8clamp.h>
#include <qnnpack/u8maxpool.h>
#include <qnnpack/u8rmax.h>
#include <qnnpack/x8zip.h>

#include "avgpool-microkernel-tester.h"
#include "clamp-microkernel-tester.h"
#include "dwconv-microkernel-tester.h"
#include "gavgpool-microkernel-tester.h"
#include "gemm-microkernel-tester.h"
#include "maxpool-microkernel-tester.h"
#include "rmax-microkernel-tester.h"
#include "vadd-microkernel-tester.h"
#include "zip-microkernel-tester.h"

TEST(Q8GEMM_4x8__SCALAR, subtiles_k_and_strides)
{
  for (size_t k = 1; k <= 15; k += 7) {
    for (size_t m = 1; m <= 4; m++) {
      for (size_t n = 1; n <= 8; n++) {
        if (k == 1 && n == 1) {
          continue;
        }
        GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(m).n(n).k(k).aStride(19).cStride(13).iterations(3).test(
            q8gemm_ukernel_4x8__scalar);
      }
    }
  }
}

TEST(Q8GEMM_4x8__SCALAR, zero_points_and_clamping)
{
  GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(4).n(8).k(17).aZeroPoint(255).bZeroPoint(0).qmin(64).qmax(192).test(
      q8gemm_ukernel_4x8__scalar);
}

TEST(Q8CONV_4x8__SCALAR, subtiles_kernel_elements_and_strides)
{
  for (size_t m = 1; m <= 4; m++) {
    for (size_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester().mr(4).nr(8).np(8).kr(1).m(m).n(n).k(11).ks(3).aStride(17).cStride(13).iterations(3).test(
          q8conv_ukernel_4x8__scalar);
    }
  }
}

TEST(Q8CONV_4x8__SCALAR, zero_points_and_clamping)
{
  GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .np(8)
      .kr(1)
      .m(4)
      .n(8)
      .k(13)
      .ks(9)
      .aZeroPoint(0)
      .bZeroPoint(255)
      .qmin(64)
      .qmax(192)
      .test(q8conv_ukernel_4x8__scalar);
}

TEST(Q8DWCONV_UP8x9__SCALAR, channels_width_strides_and_quantization)
{
  for (uint32_t channels : {1, 7, 8, 9, 16, 31}) {
    DWConvMicrokernelTester()
        .kernelHeight(3)
        .kernelWidth(3)
        .cr(8)
        .channels(channels)
        .width(5)
        .inputStride(channels + 3)
        .outputStride(channels + 5)
        .inputZeroPoint(255)
        .kernelZeroPoint(0)
        .qmin(64)
        .qmax(192)
        .iterations(3)
        .test(q8dwconv_ukernel_up8x9__scalar);
  }
}

TEST(Q8DWCONV_MP8x25__SCALAR, channels_width_strides_and_quantization)
{
  for (uint32_t channels : {1, 7, 8, 9, 16, 31}) {
    DWConvMicrokernelTester()
        .kernelHeight(5)
        .kernelWidth(5)
        .cr(8)
        .channels(channels)
        .width(3)
        .inputStride(channels + 3)
        .outputStride(channels + 5)
        .inputZeroPoint(0)
        .kernelZeroPoint(255)
        .qmin(64)
        .qmax(192)
        .iterations(3)
        .test(q8dwconv_ukernel_mp8x25__scalar);
  }
}

TEST(Q8VADD__SCALAR, lengths_quantization_and_inplace)
{
  for (size_t n : {1, 7, 8, 31, 128}) {
    VAddMicrokernelTester()
        .n(n)
        .aScale(0.75f)
        .bScale(1.25f)
        .yScale(2.0f)
        .aZeroPoint(0)
        .bZeroPoint(255)
        .yZeroPoint(127)
        .qmin(64)
        .qmax(192)
        .iterations(3)
        .test(q8vadd_ukernel__scalar);
    VAddMicrokernelTester().n(n).inplaceA(true).iterations(3).test(q8vadd_ukernel__scalar);
    VAddMicrokernelTester().n(n).inplaceB(true).iterations(3).test(q8vadd_ukernel__scalar);
  }
}

TEST(Q8AVGPOOL_UP8xM__SCALAR, channels_rows_and_strides)
{
  for (size_t kc : {1, 7}) {
    AvgPoolMicrokernelTester()
        .n(3)
        .kh(3)
        .kw(5)
        .kc(kc)
        .kr(8)
        .xStride(kc + 3)
        .yStride(kc + 5)
        .yMin(64)
        .yMax(192)
        .iterations(3)
        .test(q8avgpool_ukernel_up8xm__scalar);
  }
}

TEST(Q8AVGPOOL_UP8x9__SCALAR, channels_pooling_elements_and_strides)
{
  for (size_t kc : {8, 13, 24}) {
    for (size_t ks = 1; ks <= 9; ks++) {
      AvgPoolMicrokernelTester()
          .n(3)
          .kh(ks)
          .kw(1)
          .mr(9)
          .qr(8)
          .kc(kc)
          .kr(8)
          .xStride(kc + 3)
          .yStride(kc + 5)
          .yMin(64)
          .yMax(192)
          .iterations(3)
          .test(q8avgpool_ukernel_up8x9__scalar);
    }
  }
}

TEST(Q8AVGPOOL_MP8x9P8Q__SCALAR, channels_pooling_elements_and_strides)
{
  for (size_t kc : {8, 13, 24}) {
    for (size_t ks : {10, 16, 17, 25}) {
      AvgPoolMicrokernelTester()
          .n(3)
          .kh(ks)
          .kw(1)
          .mr(9)
          .qr(8)
          .kc(kc)
          .kr(8)
          .xStride(kc + 3)
          .yStride(kc + 5)
          .yMin(64)
          .yMax(192)
          .iterations(3)
          .test(q8avgpool_ukernel_mp8x9p8q__scalar);
    }
  }
}

TEST(Q8GAVGPOOL_UP8xM__SCALAR, rows_channels_and_stride)
{
  for (size_t m : {1, 7, 13}) {
    for (size_t n : {1, 7}) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(n + 3).yMin(64).yMax(192).iterations(3).test(
          q8gavgpool_ukernel_up8xm__scalar);
    }
  }
}

TEST(Q8GAVGPOOL_UP8x7__SCALAR, rows_channels_and_stride)
{
  for (size_t m : {1, 7}) {
    for (size_t n : {8, 13, 24}) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(n + 3).yMin(64).yMax(192).iterations(3).test(
          q8gavgpool_ukernel_up8x7__scalar);
    }
  }
}

TEST(Q8GAVGPOOL_MP8x7P7Q__SCALAR, rows_channels_and_stride)
{
  for (size_t m : {8, 13, 29}) {
    for (size_t n : {8, 13, 24}) {
      GAvgPoolMicrokernelTester().m(m).n(n).nr(8).xStride(n + 3).yMin(64).yMax(192).iterations(3).test(
          q8gavgpool_ukernel_mp8x7p7q__scalar);
    }
  }
}

TEST(U8MAXPOOL_SUB16__SCALAR, channels_rows_and_strides)
{
  for (size_t kc : {1, 7, 15}) {
    MaxPoolMicrokernelTester()
        .n(3)
        .kh(3)
        .kw(3)
        .kc(kc)
        .kr(16)
        .xStride(kc + 3)
        .yStride(kc + 5)
        .qmin(64)
        .qmax(192)
        .iterations(3)
        .test(u8maxpool_ukernel_sub16__scalar);
  }
}

TEST(U8MAXPOOL_16x9P8Q__SCALAR, channels_pooling_elements_and_strides)
{
  for (size_t kc : {16, 23, 32}) {
    for (size_t ks : {1, 8, 9, 10, 16, 17, 25}) {
      MaxPoolMicrokernelTester()
          .n(3)
          .kh(ks)
          .kw(1)
          .mr(9)
          .qr(8)
          .kc(kc)
          .kr(16)
          .xStride(kc + 3)
          .yStride(kc + 5)
          .qmin(64)
          .qmax(192)
          .iterations(3)
          .test(u8maxpool_ukernel_16x9p8q__scalar);
    }
  }
}

TEST(U8CLAMP__SCALAR, lengths_clamping_and_inplace)
{
  for (size_t n : {1, 7, 8, 31, 128}) {
    ClampMicrokernelTester().n(n).qmin(64).qmax(192).iterations(3).test(u8clamp_ukernel__scalar);
    ClampMicrokernelTester().n(n).inplace(true).qmin(64).qmax(192).iterations(3).test(u8clamp_ukernel__scalar);
  }
}

TEST(U8RMAX__SCALAR, lengths)
{
  for (size_t n : {1, 7, 8, 31, 128}) {
    RMaxMicrokernelTester().n(n).iterations(3).test(u8rmax_ukernel__scalar);
  }
}

TEST(X8ZIP__SCALAR, fixed_and_variable_groups)
{
  for (size_t n : {1, 7, 8, 31}) {
    ZipMicrokernelTester().n(n).g(2).iterations(3).test(qnnp_x8zip_x2__scalar);
    ZipMicrokernelTester().n(n).g(3).iterations(3).test(qnnp_x8zip_x3__scalar);
    ZipMicrokernelTester().n(n).g(4).iterations(3).test(qnnp_x8zip_x4__scalar);
    for (size_t groups : {4, 5, 9}) {
      ZipMicrokernelTester().n(n).g(groups).iterations(3).test(qnnp_x8zip_xm__scalar);
    }
  }
}
