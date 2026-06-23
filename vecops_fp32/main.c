#include "ssve.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static uint64_t get_time_unit()
{
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static uint64_t get_time_count()
{
    uint64_t stamp;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(stamp));
    return stamp;
}

static uint64_t get_time_interval_ns(const uint64_t t1, const uint64_t t2, const uint64_t unit)
{
    assert(t2 >= t1);
    return (t2 - t1) * 1000000000 / unit;
}

static void fill_vec(float *vec, size_t count, float weight)
{
    for (size_t i = 0; i < count; i++)
        vec[i] = weight * (i + 1);
}

static void print_choice_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <choice> <n> <iter>\n", prog);
    fprintf(stderr, "choice=1  -> mul_cf32\n");
    fprintf(stderr, "choice=2  -> power_cf32\n");
    fprintf(stderr, "choice=3  -> conj_scale_cf32\n");
    fprintf(stderr, "choice=4  -> dot_cf32\n");
    fprintf(stderr, "choice=5  -> conj_mul_cf32\n");
    fprintf(stderr, "choice=6  -> conj_dot_cf32\n");
    fprintf(stderr, "choice=7  -> mul_f32\n");
    fprintf(stderr, "choice=8  -> scale_f32\n");
    fprintf(stderr, "choice=9  -> dot_f32\n");
    fprintf(stderr, "choice=10 -> add_f32\n");
    fprintf(stderr, "Note: n is passed directly to the selected ssve kernel.\n");
}

static const char *run_ssve_choice(size_t choice, const float *a, const float *b, float *c,
                                   uint64_t n, float scale)
{
    switch (choice) {
    case 1:
        mul_cf32(a, b, c, n);
        return "mul_cf32";
    case 2:
        power_cf32(a, c, n);
        return "power_cf32";
    case 3:
        conj_scale_cf32(a, scale, c, n);
        return "conj_scale_cf32";
    case 4:
        dot_cf32(a, b, c, n);
        return "dot_cf32";
    case 5:
        conj_mul_cf32(a, b, c, n);
        return "conj_mul_cf32";
    case 6:
        conj_dot_cf32(a, b, c, n);
        return "conj_dot_cf32";
    case 7:
        mul_f32(a, b, c, n);
        return "mul_f32";
    case 8:
        scale_f32(a, scale, c, n);
        return "scale_f32";
    case 9:
        dot_f32(a, b, c, n);
        return "dot_f32";
    case 10:
        add_f32(a, b, c, n);
        return "add_f32";
    default:
        return NULL;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4) {
        fprintf(stderr, "Error argc!\n");
        print_choice_usage(argv[0]);
        return 1;
    }
    size_t choice = atoi(argv[1]), n = atoi(argv[2]), iter = atoi(argv[3]);
    if (n == 0 || iter == 0) {
        fprintf(stderr, "Error n or iter!\n");
        return 1;
    }
    float *a = NULL, *b = NULL, *c = NULL;
    a = (float *)malloc(sizeof(float) * n * 2);
    b = (float *)malloc(sizeof(float) * n * 2);
    c = (float *)malloc(sizeof(float) * n * 2);
    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "malloc failed!\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }
    fill_vec(a, n * 2, 0.1f);
    fill_vec(b, n * 2, 0.2f);
    fill_vec(c, n * 2, 0.0f);

    const float scale = 0.75f;
    uint64_t unit = get_time_unit();
    uint64_t t1 = 0;
    uint64_t t2 = 0;
    uint64_t elapsed_ns = 0;

    if (run_ssve_choice(choice, a, b, c, (uint64_t)n, scale) == NULL) {
        fprintf(stderr, "Unsupported choice: %zu\n", choice);
        print_choice_usage(argv[0]);
        free(a);
        free(b);
        free(c);
        return 1;
    }

    fill_vec(c, n * 2, 0.0f);
    t1 = get_time_count();
    for (size_t i = 0; i < iter; i++) {
        (void)run_ssve_choice(choice, a, b, c, (uint64_t)n, scale);
    }
    t2 = get_time_count();
    elapsed_ns = get_time_interval_ns(t1, t2, unit);

    printf("%llu\n", (unsigned long long)elapsed_ns);

    free(a);
    free(b);
    free(c);
    return 0;
}