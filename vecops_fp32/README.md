# Arm Streaming SVE (SSVE) Vector Operations

## Vector operations

The `ssve.h` header contains a set of basic vector operations implemented with Arm SSVE intrinsics. The kernels support two data formats, `float32` and `complex float32`. In the function names, `f32` means `float32` while `cf32` means `complex float32`. The table below summarizes the available operations.

| SSVE routine | Notes |
| --- | --- |
| `mul_cf32` | Complex element-wise multiply. |
| `power_cf32` | Outputs L2 norm for each complex element. |
| `conj_scale_cf32` | Applies complex conjugation and then scales by a real scalar. |
| `dot_cf32` | Complex inner product without conjugation. |
| `conj_mul_cf32` | Multiplies `conj(a)` by `b` element-wise. |
| `conj_dot_cf32` | Conjugate complex dot product. |
| `mul_f32` | Real element-wise multiply. |
| `scale_f32` | Scalar multiply operation. |
| `dot_f32` | Real dot product reduced into `c[0]`. |
| `add_f32` | Element-wise addition kernel. |

Important: the code in this repository is written under a fixed **512-bit** SSVE assumption. The blocking factors, tuple sizes, and tail-handling logic in `ssve.h`, as well as the way `main.c` is used for measurement, are documented with that assumption in mind. This is not presented as a vector-length-agnostic implementation.

The notes below describe the intended correspondence assuming a **512-bit** SSVE vector length, which means:

- One SSVE `float32` vector contains 16 lanes.
- `svld1_f32_x4` / `svst1_f32_x4` cover 64 `float` values per tuple.

The f32 kernels operate on `n` scalar `float32` elements, while the cf32 kernels operate on `n` scalar `complex float32` elements. The complex data is stored as interleaved `(real, imag)` pairs.

```text
[re0, im0, re1, im1, re2, im2, ...]
```

## Benchmark

`main.c` is a minimal benchmark entry point for the SSVE kernels.

- It accepts three command-line arguments: `choice`, `n`, and `iter`.
- `choice` selects one of the ten routines implemented in `ssve.h`.
- `n` is passed directly to the selected routine, so its meaning depends on whether the routine is operating on real or complex data.
- `iter` controls how many times the selected routine is executed for timing.
- The program currently prints only the total elapsed time in nanoseconds.

The benchmark maps `choice` to kernels as follows:

| `choice` | Kernel |
| --- | --- |
| `1` | `mul_cf32` |
| `2` | `power_cf32` |
| `3` | `conj_scale_cf32` |
| `4` | `dot_cf32` |
| `5` | `conj_mul_cf32` |
| `6` | `conj_dot_cf32` |
| `7` | `mul_f32` |
| `8` | `scale_f32` |
| `9` | `dot_f32` |
| `10` | `add_f32` |

The program runs the specified kernel `iter` times and then prints the total duration in `ns`.

## Build instructions

Build from the `vecops_fp32` directory and pass the compiler explicitly via `CC`.

The generated binary is intended for AArch64 targets that support the SME2 feature enabled by the Makefile flags.

```sh
make CC=clang
```

The Makefile uses the following flags:

```text
-O3 -Wall -march=armv9.2-a+sme2+sve+sve2
```

This produces the benchmark binary `vecops_fp32`.

To run the benchmark:

```sh
./vecops_fp32 <choice> <n> <iter>
```

Example:

```sh
./vecops_fp32 1 512 1000
```

To remove the generated binary:

```sh
make clean CC=clang
```

## Numerical Behavior

- The complex SSVE routines use `FCMLA`-based sequences, so their last-bit results can differ slightly from implementations with non-SIMD instructions.
- Reduction routines can also differ slightly because vector code changes the accumulation order compared with implementations with non-SIMD instructions.
