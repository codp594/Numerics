#ifndef __SSVE_H__
#define __SSVE_H__

#include <arm_sve.h>
#include <stdint.h>

void mul_cf32(const float *restrict a, const float *restrict b, float *restrict c,
              const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 32;
    svcount_t pg = svwhilelt_c32(0ULL, 2 * n, 4);
    svbool_t pt = svptrue_b32();
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32x4_t vb = svld1_f32_x4(pg, b);
        svfloat32_t vc1, vc2, vc3, vc4;
        vc1 = svdup_n_f32(0);
        vc2 = svdup_n_f32(0);
        vc3 = svdup_n_f32(0);
        vc4 = svdup_n_f32(0);
        svfloat32_t va1 = svget4_f32(va, 0), va2 = svget4_f32(va, 1), va3 = svget4_f32(va, 2),
                    va4 = svget4_f32(va, 3);
        svfloat32_t vb1 = svget4_f32(vb, 0), vb2 = svget4_f32(vb, 1), vb3 = svget4_f32(vb, 2),
                    vb4 = svget4_f32(vb, 3);
        vc1 = svcmla_f32_x(pt, vc1, va1, vb1, 0);
        vc2 = svcmla_f32_x(pt, vc2, va2, vb2, 0);
        vc3 = svcmla_f32_x(pt, vc3, va3, vb3, 0);
        vc4 = svcmla_f32_x(pt, vc4, va4, vb4, 0);
        vc1 = svcmla_f32_x(pt, vc1, va1, vb1, 90);
        vc2 = svcmla_f32_x(pt, vc2, va2, vb2, 90);
        vc3 = svcmla_f32_x(pt, vc3, va3, vb3, 90);
        vc4 = svcmla_f32_x(pt, vc4, va4, vb4, 90);
        svfloat32x4_t vc = svcreate4_f32(vc1, vc2, vc3, vc4);
        svst1_f32_x4(pg, c, vc);
        a += 64;
        b += 64;
        c += 64;
    }
    const uint64_t remain = (n & 31) / 8 + ((n & 7) ? 1 : 0);
    for (uint64_t i = 0; i < remain; i++) {
        svbool_t pb = svwhilelt_b32(16 * i, (n & 31) * 2);
        svfloat32_t va = svld1_f32(pb, a);
        svfloat32_t vb = svld1_f32(pb, b);
        svfloat32_t vc = svdup_n_f32(0);
        vc = svcmla_f32_x(pb, vc, va, vb, 0);
        vc = svcmla_f32_x(pb, vc, va, vb, 90);
        svst1_f32(pb, c, vc);
        a += 16;
        b += 16;
        c += 16;
    }
}

void power_cf32(const float *restrict a, float *restrict c, const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 64;
    svbool_t pt = svptrue_b32();
    svcount_t ptc = svptrue_c32();
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x2_t v0 = svld2_f32(pt, a);
        svfloat32x2_t v1 = svld2_f32(pt, a + 32);
        svfloat32x2_t v2 = svld2_f32(pt, a + 64);
        svfloat32x2_t v3 = svld2_f32(pt, a + 96);
        svfloat32_t va0 = svget2_f32(v0, 0), va1 = svget2_f32(v0, 1);
        svfloat32_t va2 = svget2_f32(v1, 0), va3 = svget2_f32(v1, 1);
        svfloat32_t va4 = svget2_f32(v2, 0), va5 = svget2_f32(v2, 1);
        svfloat32_t va6 = svget2_f32(v3, 0), va7 = svget2_f32(v3, 1);
        svfloat32_t r0 = svmul_f32_x(pt, va0, va0);
        svfloat32_t r1 = svmul_f32_x(pt, va2, va2);
        svfloat32_t r2 = svmul_f32_x(pt, va4, va4);
        svfloat32_t r3 = svmul_f32_x(pt, va6, va6);
        r0 = svmla_f32_x(pt, r0, va1, va1);
        r1 = svmla_f32_x(pt, r1, va3, va3);
        r2 = svmla_f32_x(pt, r2, va5, va5);
        r3 = svmla_f32_x(pt, r3, va7, va7);
        a += 128;
        svst1_f32_x4(ptc, c, svcreate4_f32(r0, r1, r2, r3));
        c += 64;
    }
    if (n & 63) {
        svbool_t p0 = svwhilelt_b32(0ULL, (n & 63));
        svbool_t p1 = svwhilelt_b32(16ULL, (n & 63));
        svbool_t p2 = svwhilelt_b32(32ULL, (n & 63));
        svbool_t p3 = svwhilelt_b32(48ULL, (n & 63));
        svcount_t pg = svwhilelt_c32(0ULL, (n & 63), 4);
        svfloat32x2_t v0 = svld2_f32(p0, a);
        svfloat32x2_t v1 = svld2_f32(p1, a + 32);
        svfloat32x2_t v2 = svld2_f32(p2, a + 64);
        svfloat32x2_t v3 = svld2_f32(p3, a + 96);
        svfloat32_t va0 = svget2_f32(v0, 0), va1 = svget2_f32(v0, 1);
        svfloat32_t va2 = svget2_f32(v1, 0), va3 = svget2_f32(v1, 1);
        svfloat32_t va4 = svget2_f32(v2, 0), va5 = svget2_f32(v2, 1);
        svfloat32_t va6 = svget2_f32(v3, 0), va7 = svget2_f32(v3, 1);
        svfloat32_t r0 = svmul_f32_x(p0, va0, va0);
        svfloat32_t r1 = svmul_f32_x(p1, va2, va2);
        svfloat32_t r2 = svmul_f32_x(p2, va4, va4);
        svfloat32_t r3 = svmul_f32_x(p3, va6, va6);
        r0 = svmla_f32_m(p0, r0, va1, va1);
        r1 = svmla_f32_m(p1, r1, va3, va3);
        r2 = svmla_f32_m(p2, r2, va5, va5);
        r3 = svmla_f32_m(p3, r3, va7, va7);
        svst1_f32_x4(pg, c, svcreate4_f32(r0, r1, r2, r3));
    }
}

void scale_f32(const float *restrict a, const float scale, float *restrict c,
               const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 128;
    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    svfloat32_t vscale = svdup_n_f32(scale);
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(ptc, a);
        svfloat32x4_t va1 = svld1_f32_x4(ptc, a + 64);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vc0 = svmul_f32_x(pt, va00, vscale);
        svfloat32_t vc1 = svmul_f32_x(pt, va01, vscale);
        svfloat32_t vc2 = svmul_f32_x(pt, va02, vscale);
        svfloat32_t vc3 = svmul_f32_x(pt, va03, vscale);
        svfloat32_t vc4 = svmul_f32_x(pt, va10, vscale);
        svfloat32_t vc5 = svmul_f32_x(pt, va11, vscale);
        svfloat32_t vc6 = svmul_f32_x(pt, va12, vscale);
        svfloat32_t vc7 = svmul_f32_x(pt, va13, vscale);
        svfloat32x4_t vc_0 = svcreate4_f32(vc0, vc1, vc2, vc3);
        svst1_f32_x4(ptc, c, vc_0);
        a += 128;
        svst1_f32(pt, c + 64, vc4);
        svst1_f32(pt, c + 80, vc5);
        svst1_f32(pt, c + 96, vc6);
        svst1_f32(pt, c + 112, vc7);
        c += 128;
    }
    for (uint64_t i = 0; i < (n & 127); i += 64) {
        svcount_t pg = svwhilelt_c32(i, n & 127, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vc0 = svmul_f32_x(pt, va0, vscale);
        svfloat32_t vc1 = svmul_f32_x(pt, va1, vscale);
        svfloat32_t vc2 = svmul_f32_x(pt, va2, vscale);
        svfloat32_t vc3 = svmul_f32_x(pt, va3, vscale);
        svfloat32x4_t vc = svcreate4_f32(vc0, vc1, vc2, vc3);
        a += 64;
        svst1_f32_x4(pg, c, vc);
        c += 64;
    }
}

void dot_f32(const float *restrict a, const float *restrict b, float *restrict c,
             const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 128;
    svfloat32_t vc0 = svdup_n_f32(0), vc1 = svdup_n_f32(0), vc2 = svdup_n_f32(0),
                vc3 = svdup_n_f32(0);

    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb0 = svld1_f32_x4(ptc, b);
        svfloat32x4_t va1 = svld1_f32_x4(ptc, a + 64);
        svfloat32x4_t vb1 = svld1_f32_x4(ptc, b + 64);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t vb00 = svget4_f32(vb0, 0), vb01 = svget4_f32(vb0, 1), vb02 = svget4_f32(vb0, 2),
                    vb03 = svget4_f32(vb0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vb10 = svget4_f32(vb1, 0), vb11 = svget4_f32(vb1, 1), vb12 = svget4_f32(vb1, 2),
                    vb13 = svget4_f32(vb1, 3);
        vc0 = svmla_f32_x(pt, vc0, va00, vb00);
        vc1 = svmla_f32_x(pt, vc1, va01, vb01);
        vc2 = svmla_f32_x(pt, vc2, va02, vb02);
        vc3 = svmla_f32_x(pt, vc3, va03, vb03);
        vc0 = svmla_f32_x(pt, vc0, va10, vb10);
        vc1 = svmla_f32_x(pt, vc1, va11, vb11);
        vc2 = svmla_f32_x(pt, vc2, va12, vb12);
        vc3 = svmla_f32_x(pt, vc3, va13, vb13);
        a += 128;
        b += 128;
    }
    if (n & 64) {
        svfloat32x4_t va = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb = svld1_f32_x4(ptc, b);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        vc0 = svmla_f32_x(pt, vc0, va0, vb0);
        vc1 = svmla_f32_x(pt, vc1, va1, vb1);
        vc2 = svmla_f32_x(pt, vc2, va2, vb2);
        vc3 = svmla_f32_x(pt, vc3, va3, vb3);
        a += 64;
        b += 64;
    }
    if (n & 63) {
        ptc = svwhilelt_c32(0ULL, (n & 63), 4);
        svfloat32x4_t va = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb = svld1_f32_x4(ptc, b);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        svboolx2_t p01 = svpext_lane_c32_x2(ptc, 0), p23 = svpext_lane_c32_x2(ptc, 1);
        svbool_t p0 = svget2(p01, 0), p1 = svget2(p01, 1), p2 = svget2(p23, 0), p3 = svget2(p23, 1);
        vc0 = svmla_f32_m(p0, vc0, va0, vb0);
        vc1 = svmla_f32_m(p1, vc1, va1, vb1);
        vc2 = svmla_f32_m(p2, vc2, va2, vb2);
        vc3 = svmla_f32_m(p3, vc3, va3, vb3);
    }
    svfloat32_t res_p0 = svadd_f32_x(pt, vc0, vc1);
    svfloat32_t res_p1 = svadd_f32_x(pt, vc2, vc3);
    svfloat32_t res = svadd_f32_x(pt, res_p0, res_p1);
    *c = svaddv(pt, res);
}

void add_f32(const float *restrict a, const float *restrict b, float *restrict c,
             const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 128;
    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(ptc, a);
        svfloat32x4_t va1 = svld1_f32_x4(ptc, a + 64);
        svfloat32x4_t vb0 = svld1_f32_x4(ptc, b);
        svfloat32x4_t vb1 = svld1_f32_x4(ptc, b + 64);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vb00 = svget4_f32(vb0, 0), vb01 = svget4_f32(vb0, 1), vb02 = svget4_f32(vb0, 2),
                    vb03 = svget4_f32(vb0, 3);
        svfloat32_t vb10 = svget4_f32(vb1, 0), vb11 = svget4_f32(vb1, 1), vb12 = svget4_f32(vb1, 2),
                    vb13 = svget4_f32(vb1, 3);
        svfloat32_t vc00 = svadd_f32_x(pt, va00, vb00);
        svfloat32_t vc01 = svadd_f32_x(pt, va01, vb01);
        svfloat32_t vc02 = svadd_f32_x(pt, va02, vb02);
        svfloat32_t vc03 = svadd_f32_x(pt, va03, vb03);
        svfloat32x4_t vc0 = svcreate4_f32(vc00, vc01, vc02, vc03);
        svfloat32_t vc10 = svadd_f32_x(pt, va10, vb10);
        svfloat32_t vc11 = svadd_f32_x(pt, va11, vb11);
        svfloat32_t vc12 = svadd_f32_x(pt, va12, vb12);
        svfloat32_t vc13 = svadd_f32_x(pt, va13, vb13);
        svfloat32x4_t vc1 = svcreate4_f32(vc10, vc11, vc12, vc13);
        a += 128;
        b += 128;
        svst1_f32_x4(ptc, c, vc0);
        svst1_f32_x4(ptc, c + 64, vc1);
        c += 128;
    }
    for (uint64_t i = 0; i < (n & 127); i += 64) {
        svcount_t pg = svwhilelt_c32(i, n & 127, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32x4_t vb = svld1_f32_x4(pg, b);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        svfloat32_t vc0 = svadd_f32_x(pt, va0, vb0);
        svfloat32_t vc1 = svadd_f32_x(pt, va1, vb1);
        svfloat32_t vc2 = svadd_f32_x(pt, va2, vb2);
        svfloat32_t vc3 = svadd_f32_x(pt, va3, vb3);
        svfloat32x4_t vc = svcreate4_f32(vc0, vc1, vc2, vc3);
        a += 64;
        b += 64;
        svst1_f32_x4(pg, c, vc);
        c += 64;
    }
}

void conj_scale_cf32(const float *restrict a, const float scale, float *restrict c,
                     const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 32;
    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    svfloat32_t vs1 = svdup_n_f32(scale);
    svfloat32_t vs2 = svneg_f32_x(pt, vs1);
    svfloat32_t vscale = svzip1_f32(vs1, vs2);
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va = svld1_f32_x4(ptc, a);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vc0 = svmul_f32_x(pt, va0, vscale);
        svfloat32_t vc1 = svmul_f32_x(pt, va1, vscale);
        svfloat32_t vc2 = svmul_f32_x(pt, va2, vscale);
        svfloat32_t vc3 = svmul_f32_x(pt, va3, vscale);
        a += 64;
        svfloat32x4_t vc = svcreate4_f32(vc0, vc1, vc2, vc3);
        svst1_f32_x4(ptc, c, vc);
        c += 64;
    }
    if (n & 31) {
        svcount_t pg = svwhilelt_c32(0ULL, (n & 31) * 2, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vc0 = svmul_f32_x(pt, va0, vscale);
        svfloat32_t vc1 = svmul_f32_x(pt, va1, vscale);
        svfloat32_t vc2 = svmul_f32_x(pt, va2, vscale);
        svfloat32_t vc3 = svmul_f32_x(pt, va3, vscale);
        svfloat32x4_t vc = svcreate4_f32(vc0, vc1, vc2, vc3);
        svst1_f32_x4(pg, c, vc);
    }
}

void dot_cf32(const float *restrict a, const float *restrict b, float *restrict c,
              const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 64;
    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    svfloat32_t vc0 = svdup_n_f32(0), vc1 = svdup_n_f32(0), vc2 = svdup_n_f32(0),
                vc3 = svdup_n_f32(0);
    svfloat32_t vc4 = svdup_n_f32(0), vc5 = svdup_n_f32(0), vc6 = svdup_n_f32(0),
                vc7 = svdup_n_f32(0);
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb0 = svld1_f32_x4(ptc, b);
        svfloat32x4_t va1 = svld1_f32_x4(ptc, a + 64);
        svfloat32x4_t vb1 = svld1_f32_x4(ptc, b + 64);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t vb00 = svget4_f32(vb0, 0), vb01 = svget4_f32(vb0, 1), vb02 = svget4_f32(vb0, 2),
                    vb03 = svget4_f32(vb0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vb10 = svget4_f32(vb1, 0), vb11 = svget4_f32(vb1, 1), vb12 = svget4_f32(vb1, 2),
                    vb13 = svget4_f32(vb1, 3);
        vc0 = svcmla_f32_x(pt, vc0, va00, vb00, 0);
        vc1 = svcmla_f32_x(pt, vc1, va01, vb01, 0);
        vc2 = svcmla_f32_x(pt, vc2, va02, vb02, 0);
        vc3 = svcmla_f32_x(pt, vc3, va03, vb03, 0);
        vc4 = svcmla_f32_x(pt, vc4, va10, vb10, 0);
        vc5 = svcmla_f32_x(pt, vc5, va11, vb11, 0);
        vc6 = svcmla_f32_x(pt, vc6, va12, vb12, 0);
        vc7 = svcmla_f32_x(pt, vc7, va13, vb13, 0);
        vc0 = svcmla_f32_x(pt, vc0, va00, vb00, 90);
        vc1 = svcmla_f32_x(pt, vc1, va01, vb01, 90);
        vc2 = svcmla_f32_x(pt, vc2, va02, vb02, 90);
        vc3 = svcmla_f32_x(pt, vc3, va03, vb03, 90);
        vc4 = svcmla_f32_x(pt, vc4, va10, vb10, 90);
        vc5 = svcmla_f32_x(pt, vc5, va11, vb11, 90);
        vc6 = svcmla_f32_x(pt, vc6, va12, vb12, 90);
        vc7 = svcmla_f32_x(pt, vc7, va13, vb13, 90);
        a += 128;
        b += 128;
    }
    if (n & 32) {
        svfloat32x4_t va = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb = svld1_f32_x4(ptc, b);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        vc0 = svcmla_f32_x(pt, vc0, va0, vb0, 0);
        vc1 = svcmla_f32_x(pt, vc1, va1, vb1, 0);
        vc2 = svcmla_f32_x(pt, vc2, va2, vb2, 0);
        vc3 = svcmla_f32_x(pt, vc3, va3, vb3, 0);
        vc0 = svcmla_f32_x(pt, vc0, va0, vb0, 90);
        vc1 = svcmla_f32_x(pt, vc1, va1, vb1, 90);
        vc2 = svcmla_f32_x(pt, vc2, va2, vb2, 90);
        vc3 = svcmla_f32_x(pt, vc3, va3, vb3, 90);
        a += 64;
        b += 64;
    }
    if (n & 31) {
        svcount_t pg = svwhilelt_c32(0ULL, (n & 31) * 2, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32x4_t vb = svld1_f32_x4(pg, b);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        svboolx2_t p01 = svpext_lane_c32_x2(pg, 0);
        svboolx2_t p23 = svpext_lane_c32_x2(pg, 1);
        svbool_t p0 = svget2(p01, 0), p1 = svget2(p01, 1), p2 = svget2(p23, 0), p3 = svget2(p23, 1);
        vc0 = svcmla_f32_m(p0, vc0, va0, vb0, 0);
        vc1 = svcmla_f32_m(p1, vc1, va1, vb1, 0);
        vc2 = svcmla_f32_m(p2, vc2, va2, vb2, 0);
        vc3 = svcmla_f32_m(p3, vc3, va3, vb3, 0);
        vc0 = svcmla_f32_m(p0, vc0, va0, vb0, 90);
        vc1 = svcmla_f32_m(p1, vc1, va1, vb1, 90);
        vc2 = svcmla_f32_m(p2, vc2, va2, vb2, 90);
        vc3 = svcmla_f32_m(p3, vc3, va3, vb3, 90);
    }
    svfloat32_t res_p0 = svadd_f32_x(pt, vc0, vc1);
    svfloat32_t res_p1 = svadd_f32_x(pt, vc2, vc3);
    svfloat32_t res_p2 = svadd_f32_x(pt, vc4, vc5);
    svfloat32_t res_p3 = svadd_f32_x(pt, vc6, vc7);
    svfloat32_t res0 = svadd_f32_x(pt, res_p0, res_p1);
    svfloat32_t res1 = svadd_f32_x(pt, res_p2, res_p3);
    svfloat32_t r0 = svuzp1_f32(res0, res1);
    svfloat32_t r1 = svuzp2_f32(res0, res1);
    *c = svaddv_f32(pt, r0);
    *(c + 1) = svaddv_f32(pt, r1);
}

void mul_f32(const float *restrict a, const float *restrict b, float *restrict c,
             const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 128;
    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb0 = svld1_f32_x4(ptc, b);
        svfloat32x4_t va1 = svld1_f32_x4(ptc, a + 64);
        svfloat32x4_t vb1 = svld1_f32_x4(ptc, b + 64);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vb00 = svget4_f32(vb0, 0), vb01 = svget4_f32(vb0, 1), vb02 = svget4_f32(vb0, 2),
                    vb03 = svget4_f32(vb0, 3);
        svfloat32_t vb10 = svget4_f32(vb1, 0), vb11 = svget4_f32(vb1, 1), vb12 = svget4_f32(vb1, 2),
                    vb13 = svget4_f32(vb1, 3);
        svfloat32_t vc00 = svmul_f32_x(pt, va00, vb00);
        svfloat32_t vc01 = svmul_f32_x(pt, va01, vb01);
        svfloat32_t vc02 = svmul_f32_x(pt, va02, vb02);
        svfloat32_t vc03 = svmul_f32_x(pt, va03, vb03);
        svfloat32_t vc10 = svmul_f32_x(pt, va10, vb10);
        svfloat32_t vc11 = svmul_f32_x(pt, va11, vb11);
        svfloat32_t vc12 = svmul_f32_x(pt, va12, vb12);
        svfloat32_t vc13 = svmul_f32_x(pt, va13, vb13);
        a += 128;
        b += 128;
        svfloat32x4_t vc0 = svcreate4_f32(vc00, vc01, vc02, vc03);
        svst1_f32_x4(ptc, c, vc0);
        svfloat32x4_t vc1 = svcreate4_f32(vc10, vc11, vc12, vc13);
        svst1_f32_x4(ptc, c + 64, vc1);
        c += 128;
    }
    for (uint64_t i = 0; i < (n & 127); i += 64) {
        svcount_t pg = svwhilelt_c32(i, n & 127, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32x4_t vb = svld1_f32_x4(pg, b);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        svfloat32_t vc0 = svmul_f32_x(pt, va0, vb0);
        svfloat32_t vc1 = svmul_f32_x(pt, va1, vb1);
        svfloat32_t vc2 = svmul_f32_x(pt, va2, vb2);
        svfloat32_t vc3 = svmul_f32_x(pt, va3, vb3);
        svfloat32x4_t vc = svcreate4_f32(vc0, vc1, vc2, vc3);
        a += 64;
        b += 64;
        svst1_f32_x4(pg, c, vc);
        c += 64;
    }
}

void conj_mul_cf32(const float *restrict a, const float *restrict b, float *restrict c,
                   const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 64;
    svcount_t pg = svptrue_c32();
    svbool_t pt = svptrue_b32();
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(pg, a);
        svfloat32x4_t vb0 = svld1_f32_x4(pg, b);
        svfloat32_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7;
        svfloat32x4_t va1 = svld1_f32_x4(pg, a + 64);
        svfloat32x4_t vb1 = svld1_f32_x4(pg, b + 64);
        vc0 = svdup_n_f32(0);
        vc1 = svdup_n_f32(0);
        vc2 = svdup_n_f32(0);
        vc3 = svdup_n_f32(0);
        vc4 = svdup_n_f32(0);
        vc5 = svdup_n_f32(0);
        vc6 = svdup_n_f32(0);
        vc7 = svdup_n_f32(0);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vb00 = svget4_f32(vb0, 0), vb01 = svget4_f32(vb0, 1), vb02 = svget4_f32(vb0, 2),
                    vb03 = svget4_f32(vb0, 3);
        svfloat32_t vb10 = svget4_f32(vb1, 0), vb11 = svget4_f32(vb1, 1), vb12 = svget4_f32(vb1, 2),
                    vb13 = svget4_f32(vb1, 3);
        vc0 = svcmla_f32_x(pt, vc0, vb00, va00, 0);
        vc1 = svcmla_f32_x(pt, vc1, vb01, va01, 0);
        vc2 = svcmla_f32_x(pt, vc2, vb02, va02, 0);
        vc3 = svcmla_f32_x(pt, vc3, vb03, va03, 0);
        vc4 = svcmla_f32_x(pt, vc4, vb10, va10, 0);
        vc5 = svcmla_f32_x(pt, vc5, vb11, va11, 0);
        vc6 = svcmla_f32_x(pt, vc6, vb12, va12, 0);
        vc7 = svcmla_f32_x(pt, vc7, vb13, va13, 0);
        vc0 = svcmla_f32_x(pt, vc0, vb00, va00, 270);
        vc1 = svcmla_f32_x(pt, vc1, vb01, va01, 270);
        vc2 = svcmla_f32_x(pt, vc2, vb02, va02, 270);
        vc3 = svcmla_f32_x(pt, vc3, vb03, va03, 270);
        svfloat32x4_t vc_p0 = svcreate4_f32(vc0, vc1, vc2, vc3);
        svst1_f32_x4(pg, c, vc_p0);
        vc4 = svcmla_f32_x(pt, vc4, vb10, va10, 270);
        vc5 = svcmla_f32_x(pt, vc5, vb11, va11, 270);
        vc6 = svcmla_f32_x(pt, vc6, vb12, va12, 270);
        vc7 = svcmla_f32_x(pt, vc7, vb13, va13, 270);
        svfloat32x4_t vc_p1 = svcreate4_f32(vc4, vc5, vc6, vc7);
        a += 128;
        b += 128;
        svst1_f32_x4(pg, c + 64, vc_p1);
        c += 128;
    }
    for (uint64_t i = 0; i < (n & 63); i += 32) {
        svcount_t pg = svwhilelt_c32(i * 2, (n & 63) * 2, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32x4_t vb = svld1_f32_x4(pg, b);
        svfloat32_t vc0, vc1, vc2, vc3;
        vc0 = svdup_n_f32(0);
        vc1 = svdup_n_f32(0);
        vc2 = svdup_n_f32(0);
        vc3 = svdup_n_f32(0);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        vc0 = svcmla_f32_x(pt, vc0, vb0, va0, 0);
        vc1 = svcmla_f32_x(pt, vc1, vb1, va1, 0);
        vc2 = svcmla_f32_x(pt, vc2, vb2, va2, 0);
        vc3 = svcmla_f32_x(pt, vc3, vb3, va3, 0);
        vc0 = svcmla_f32_x(pt, vc0, vb0, va0, 270);
        vc1 = svcmla_f32_x(pt, vc1, vb1, va1, 270);
        vc2 = svcmla_f32_x(pt, vc2, vb2, va2, 270);
        vc3 = svcmla_f32_x(pt, vc3, vb3, va3, 270);
        a += 64;
        b += 64;
        svfloat32x4_t vc = svcreate4_f32(vc0, vc1, vc2, vc3);
        svst1_f32_x4(pg, c, vc);
        c += 64;
    }
}

void conj_dot_cf32(const float *restrict a, const float *restrict b, float *restrict c,
                   const uint64_t n) __arm_streaming
{
    const uint64_t iter = n / 64;
    svcount_t ptc = svptrue_c32();
    svbool_t pt = svptrue_b32();
    svfloat32_t vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7;
    vc0 = svdup_n_f32(0);
    vc1 = svdup_n_f32(0);
    vc2 = svdup_n_f32(0);
    vc3 = svdup_n_f32(0);
    vc4 = svdup_n_f32(0);
    vc5 = svdup_n_f32(0);
    vc6 = svdup_n_f32(0);
    vc7 = svdup_n_f32(0);
    for (uint64_t i = 0; i < iter; i++) {
        svfloat32x4_t va0 = svld1_f32_x4(ptc, a);
        svfloat32x4_t vb0 = svld1_f32_x4(ptc, b);
        svfloat32x4_t va1 = svld1_f32_x4(ptc, a + 64);
        svfloat32x4_t vb1 = svld1_f32_x4(ptc, b + 64);
        svfloat32_t va00 = svget4_f32(va0, 0), va01 = svget4_f32(va0, 1), va02 = svget4_f32(va0, 2),
                    va03 = svget4_f32(va0, 3);
        svfloat32_t va10 = svget4_f32(va1, 0), va11 = svget4_f32(va1, 1), va12 = svget4_f32(va1, 2),
                    va13 = svget4_f32(va1, 3);
        svfloat32_t vb00 = svget4_f32(vb0, 0), vb01 = svget4_f32(vb0, 1), vb02 = svget4_f32(vb0, 2),
                    vb03 = svget4_f32(vb0, 3);
        svfloat32_t vb10 = svget4_f32(vb1, 0), vb11 = svget4_f32(vb1, 1), vb12 = svget4_f32(vb1, 2),
                    vb13 = svget4_f32(vb1, 3);
        vc0 = svcmla_f32_x(pt, vc0, vb00, va00, 0);
        vc1 = svcmla_f32_x(pt, vc1, vb01, va01, 0);
        vc2 = svcmla_f32_x(pt, vc2, vb02, va02, 0);
        vc3 = svcmla_f32_x(pt, vc3, vb03, va03, 0);
        vc4 = svcmla_f32_x(pt, vc4, vb10, va10, 0);
        vc5 = svcmla_f32_x(pt, vc5, vb11, va11, 0);
        vc6 = svcmla_f32_x(pt, vc6, vb12, va12, 0);
        vc7 = svcmla_f32_x(pt, vc7, vb13, va13, 0);
        vc0 = svcmla_f32_x(pt, vc0, vb00, va00, 270);
        vc1 = svcmla_f32_x(pt, vc1, vb01, va01, 270);
        vc2 = svcmla_f32_x(pt, vc2, vb02, va02, 270);
        vc3 = svcmla_f32_x(pt, vc3, vb03, va03, 270);
        vc4 = svcmla_f32_x(pt, vc4, vb10, va10, 270);
        vc5 = svcmla_f32_x(pt, vc5, vb11, va11, 270);
        vc6 = svcmla_f32_x(pt, vc6, vb12, va12, 270);
        vc7 = svcmla_f32_x(pt, vc7, vb13, va13, 270);
        a += 128;
        b += 128;
    }
    for (uint64_t i = 0; i < (n & 63); i += 32) {
        svcount_t pg = svwhilelt_c32(i * 2, (n & 63) * 2, 4);
        svfloat32x4_t va = svld1_f32_x4(pg, a);
        svfloat32x4_t vb = svld1_f32_x4(pg, b);
        svboolx2_t p01 = svpext_lane_c32_x2(pg, 0);
        svboolx2_t p23 = svpext_lane_c32_x2(pg, 1);
        svbool_t p0 = svget2(p01, 0), p1 = svget2(p01, 1), p2 = svget2(p23, 0), p3 = svget2(p23, 1);
        svfloat32_t va0 = svget4_f32(va, 0), va1 = svget4_f32(va, 1), va2 = svget4_f32(va, 2),
                    va3 = svget4_f32(va, 3);
        svfloat32_t vb0 = svget4_f32(vb, 0), vb1 = svget4_f32(vb, 1), vb2 = svget4_f32(vb, 2),
                    vb3 = svget4_f32(vb, 3);
        vc0 = svcmla_f32_m(p0, vc0, vb0, va0, 0);
        vc1 = svcmla_f32_m(p1, vc1, vb1, va1, 0);
        vc2 = svcmla_f32_m(p2, vc2, vb2, va2, 0);
        vc3 = svcmla_f32_m(p3, vc3, vb3, va3, 0);
        vc0 = svcmla_f32_m(p0, vc0, vb0, va0, 270);
        vc1 = svcmla_f32_m(p1, vc1, vb1, va1, 270);
        vc2 = svcmla_f32_m(p2, vc2, vb2, va2, 270);
        vc3 = svcmla_f32_m(p3, vc3, vb3, va3, 270);
        a += 64;
        b += 64;
    }
    svfloat32_t res_p0 = svadd_f32_x(pt, vc0, vc1);
    svfloat32_t res_p1 = svadd_f32_x(pt, vc2, vc3);
    svfloat32_t res_p2 = svadd_f32_x(pt, vc4, vc5);
    svfloat32_t res_p3 = svadd_f32_x(pt, vc6, vc7);
    svfloat32_t res0 = svadd_f32_x(pt, res_p0, res_p1);
    svfloat32_t res1 = svadd_f32_x(pt, res_p2, res_p3);
    svfloat32_t res_real = svuzp1_f32(res0, res1);
    svfloat32_t res_imag = svuzp2_f32(res0, res1);
    float real = svaddv_f32(pt, res_real);
    float imag = svaddv_f32(pt, res_imag);
    *c = real;
    *(c + 1) = imag;
}

#endif