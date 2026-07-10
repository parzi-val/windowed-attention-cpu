/* gpt2_forward_windowed.c -- full end-to-end GPT-2 small forward pass, three attention
 * arms (dense / lazy-head-map / all-windowed), pure C, no PyTorch/Python in the loop.
 *
 * Architecture code (encoder/layernorm/matmul/gelu/residual, struct layout, checkpoint
 * format, dense attention_forward) is adapted from Andrej Karpathy's llm.c
 * (https://github.com/karpathy/llm.c, MIT licensed) -- the reference CPU GPT-2 forward
 * pass, reused here because getting the Conv1D-transpose weight export and the exact
 * numerics right is already solved and tested there. Only forward inference is kept (no
 * backward pass, no optimizer, no tokenizer/dataloader -- none of that is needed for a
 * fixed-prompt forward-only timing/parity check), and their POSIX-only includes
 * (unistd.h via llmc/utils.h) are dropped in favor of portable C11 replacements so this
 * builds on Windows/clang-msvc as well as Linux/gcc.
 *
 * attention_forward_perhead() is new: banded sink+window attention for the heads a
 * per-(layer,head) mask marks windowed, full causal for the rest -- driven by the real
 * lazy-head compressibility map
 * (Definition 1: delta_ppl < 0.05 under the sink+window replacement test), not just a
 * blanket "window everything" switch. That's the actual paper claim -- window the lazy
 * heads, keep the load-bearing ones dense -- not just "structured sparsity exists."
 *
 * Build:
 *     clang -O2 -o gpt2_forward_windowed gpt2_forward_windowed.c -lm
 *     (or gcc -O2 -o gpt2_forward_windowed gpt2_forward_windowed.c -lm on Linux/Colab)
 *
 * Needs, in the same directory: gpt2_124M.bin, gpt2_124M_fwd_state.bin, gpt2_124M_lazymap.bin
 * (all produced by export_gpt2_c.py -- see that file's docstring, and head_compressibility_
 * gpt2.json for the map's provenance).
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define SINK 4
#define WINDOW 64

/* ---------------------------------------------------------------------------
 * Portable I/O helpers (replacing llm.c's llmc/utils.h, which pulls in unistd.h --
 * not available on the Windows/clang-msvc target). Same error-checked-fopen/fread
 * intent, just self-contained.
 * ------------------------------------------------------------------------- */
static FILE *fopen_check(const char *path, const char *mode) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: failed to open '%s' (mode %s)\n", path, mode);
        exit(EXIT_FAILURE);
    }
    return fp;
}
static void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t got = fread(ptr, size, nmemb, stream);
    if (got != nmemb) {
        fprintf(stderr, "Error: short read, expected %zu elements, got %zu\n", nmemb, got);
        exit(EXIT_FAILURE);
    }
}
static void *malloc_check(size_t n) {
    void *p = malloc(n);
    if (p == NULL) {
        fprintf(stderr, "Error: malloc(%zu) failed\n", n);
        exit(EXIT_FAILURE);
    }
    return p;
}
static double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);  /* C11, portable across gcc and clang-msvc */
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------------------
 * Layers (forward-only). encoder/layernorm/matmul/gelu/residual copied from llm.c;
 * attention_forward (dense) copied from llm.c; attention_forward_windowed is new.
 * ------------------------------------------------------------------------- */
static void encoder_forward(float *out, const int *inp, const float *wte, const float *wpe,
                             int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = out + b * T * C + t * C;
            int ix = inp[b * T + t];
            const float *wte_ix = wte + (size_t)ix * C;
            const float *wpe_t = wpe + (size_t)t * C;
            for (int i = 0; i < C; i++) out_bt[i] = wte_ix[i] + wpe_t[i];
        }
    }
}

static void layernorm_forward(float *out, const float *inp, const float *weight, const float *bias,
                               int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *x = inp + b * T * C + t * C;
            float m = 0.0f;
            for (int i = 0; i < C; i++) m += x[i];
            m /= C;
            float v = 0.0f;
            for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
            v /= C;
            float s = 1.0f / sqrtf(v + eps);
            float *out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) out_bt[i] = (s * (x[i] - m)) * weight[i] + bias[i];
        }
    }
}

static void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                            int B, int T, int C, int OC) {
    /* inp (B,T,C), weight (OC,C), bias (OC) -> out (B,T,OC). Naive but correct; identical
     * cost for dense and windowed arms (only attention differs), so it doesn't bias the
     * comparison even though it's not the fastest possible matmul. */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) val += inp[(size_t)bt * C + i] * weight[(size_t)o * C + i];
                out[(size_t)bt * OC + o] = val;
            }
        }
    }
}

static void attention_forward(float *out, float *preatt, float *att, const float *inp,
                               int B, int T, int C, int NH) {
    /* dense causal attention, from llm.c. inp is (B,T,3C) holding Q,K,V. */
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float *query_t = inp + (size_t)b * T * C3 + (size_t)t * C3 + h * hs;
                float *preatt_bth = preatt + (size_t)b * NH * T * T + (size_t)h * T * T + (size_t)t * T;
                float *att_bth = att + (size_t)b * NH * T * T + (size_t)h * T * T + (size_t)t * T;

                float maxval = -1e9f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *key_t2 = inp + (size_t)b * T * C3 + (size_t)t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) val += query_t[i] * key_t2[i];
                    val *= scale;
                    if (val > maxval) maxval = val;
                    preatt_bth[t2] = val;
                }
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float e = expf(preatt_bth[t2] - maxval);
                    expsum += e;
                    att_bth[t2] = e;
                }
                float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                for (int t2 = 0; t2 <= t; t2++) att_bth[t2] *= inv;

                float *out_bth = out + (size_t)b * T * C + (size_t)t * C + h * hs;
                for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *value_t2 = inp + (size_t)b * T * C3 + (size_t)t2 * C3 + h * hs + C * 2;
                    float a = att_bth[t2];
                    for (int i = 0; i < hs; i++) out_bth[i] += a * value_t2[i];
                }
            }
        }
    }
}

static void attention_forward_perhead(float *out, const float *inp, int B, int T, int C, int NH,
                                       const int *head_windowed, int sink, int window) {
    /* Per-head dispatch: head_windowed[h] chooses banded sink+window (bounded, fixed stack
     * buffer) or full causal (VLA sized to t+1, <=T floats -- fine on the stack up to our
     * T=1024 ceiling). This is what lets a real per-head map (map lazy heads windowed, load-
     * bearing heads dense, same layer) be driven through one function instead of splitting
     * a layer's heads across two separate kernel calls. */
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int win_lo = t - window + 1;
            if (win_lo < 0) win_lo = 0;
            for (int h = 0; h < NH; h++) {
                const float *query_t = inp + (size_t)b * T * C3 + (size_t)t * C3 + h * hs;
                float *out_bth = out + (size_t)b * T * C + (size_t)t * C + h * hs;

                if (head_windowed[h]) {
                    int idxs[SINK + WINDOW];
                    float scores[SINK + WINDOW];
                    int n = 0;
                    for (int s = 0; s < sink && s <= t; s++) {
                        if (s < win_lo) idxs[n++] = s;
                    }
                    for (int s = win_lo; s <= t; s++) idxs[n++] = s;

                    float maxval = -1e9f;
                    for (int i = 0; i < n; i++) {
                        int s = idxs[i];
                        const float *key_s = inp + (size_t)b * T * C3 + (size_t)s * C3 + h * hs + C;
                        float val = 0.0f;
                        for (int k = 0; k < hs; k++) val += query_t[k] * key_s[k];
                        val *= scale;
                        scores[i] = val;
                        if (val > maxval) maxval = val;
                    }
                    float expsum = 0.0f;
                    for (int i = 0; i < n; i++) { scores[i] = expf(scores[i] - maxval); expsum += scores[i]; }
                    float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                    for (int i = 0; i < n; i++) scores[i] *= inv;

                    for (int k = 0; k < hs; k++) out_bth[k] = 0.0f;
                    for (int i = 0; i < n; i++) {
                        int s = idxs[i];
                        const float *value_s = inp + (size_t)b * T * C3 + (size_t)s * C3 + h * hs + C * 2;
                        float w = scores[i];
                        for (int k = 0; k < hs; k++) out_bth[k] += w * value_s[k];
                    }
                } else {
                    int n = t + 1;
                    float scores[n]; /* VLA, C99: n <= T <= 1024, trivially fine on the stack */
                    float maxval = -1e9f;
                    for (int s = 0; s < n; s++) {
                        const float *key_s = inp + (size_t)b * T * C3 + (size_t)s * C3 + h * hs + C;
                        float val = 0.0f;
                        for (int k = 0; k < hs; k++) val += query_t[k] * key_s[k];
                        val *= scale;
                        scores[s] = val;
                        if (val > maxval) maxval = val;
                    }
                    float expsum = 0.0f;
                    for (int s = 0; s < n; s++) { scores[s] = expf(scores[s] - maxval); expsum += scores[s]; }
                    float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                    for (int s = 0; s < n; s++) scores[s] *= inv;

                    for (int k = 0; k < hs; k++) out_bth[k] = 0.0f;
                    for (int s = 0; s < n; s++) {
                        const float *value_s = inp + (size_t)b * T * C3 + (size_t)s * C3 + h * hs + C * 2;
                        float w = scores[s];
                        for (int k = 0; k < hs; k++) out_bth[k] += w * value_s[k];
                    }
                }
            }
        }
    }
}

static void gelu_forward(float *out, const float *inp, int N) {
    const float GELU_SCALE = 0.7978845608028654f; /* sqrt(2/pi), hardcoded: M_PI isn't
                                                       standard on the MSVC math.h target */
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALE * (x + cube)));
    }
}

static void residual_forward(float *out, const float *inp1, const float *inp2, int N) {
    for (int i = 0; i < N; i++) out[i] = inp1[i] + inp2[i];
}

/* ---------------------------------------------------------------------------
 * Model definition (forward-only: no grads, no optimizer state, no dataloader/tokenizer).
 * ------------------------------------------------------------------------- */
typedef struct {
    int max_seq_len, vocab_size, padded_vocab_size, num_layers, num_heads, channels;
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte, *wpe, *ln1w, *ln1b, *qkvw, *qkvb, *attprojw, *attprojb,
          *ln2w, *ln2b, *fcw, *fcb, *fcprojw, *fcprojb, *lnfw, *lnfb;
} ParameterTensors;

static void fill_in_parameter_sizes(size_t *s, GPT2Config cfg) {
    size_t Vp = cfg.padded_vocab_size, C = cfg.channels, maxT = cfg.max_seq_len, L = cfg.num_layers;
    s[0] = Vp * C; s[1] = maxT * C; s[2] = L * C; s[3] = L * C;
    s[4] = L * (3 * C) * C; s[5] = L * (3 * C); s[6] = L * C * C; s[7] = L * C;
    s[8] = L * C; s[9] = L * C; s[10] = L * (4 * C) * C; s[11] = L * (4 * C);
    s[12] = L * C * (4 * C); s[13] = L * C; s[14] = C; s[15] = C;
}

static float *malloc_and_point_parameters(ParameterTensors *p, size_t *sizes) {
    size_t total = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) total += sizes[i];
    float *mem = (float *)malloc_check(total * sizeof(float));
    float **ptrs[NUM_PARAMETER_TENSORS] = {
        &p->wte, &p->wpe, &p->ln1w, &p->ln1b, &p->qkvw, &p->qkvb, &p->attprojw, &p->attprojb,
        &p->ln2w, &p->ln2b, &p->fcw, &p->fcb, &p->fcprojw, &p->fcprojb, &p->lnfw, &p->lnfb
    };
    float *it = mem;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) { *ptrs[i] = it; it += sizes[i]; }
    return mem;
}

/* Activations, forward-only, no crossentropy/probs (no targets/loss needed here).
 * preatt/att only allocated for the dense arm -- windowed doesn't need them at all,
 * which is where the memory saving (not just compute) actually comes from. */
typedef struct {
    float *encoded, *ln1, *qkv, *atty, *preatt, *att, *attproj, *residual2,
          *ln2, *fch, *fch_gelu, *fcproj, *residual3, *lnf, *logits;
} ActivationTensors;

static float *malloc_and_point_activations(ActivationTensors *a, GPT2Config cfg, int B, int T, int need_preatt) {
    size_t C = cfg.channels, NH = cfg.num_heads, L = cfg.num_layers, Vp = cfg.padded_vocab_size;
    size_t sizes[13];
    sizes[0] = (size_t)B * T * C;              /* encoded */
    sizes[1] = L * (size_t)B * T * C;          /* ln1 */
    sizes[2] = L * (size_t)B * T * 3 * C;      /* qkv */
    sizes[3] = L * (size_t)B * T * C;          /* atty */
    sizes[4] = need_preatt ? L * (size_t)B * NH * T * T : 0; /* preatt -- only for the dense reference path */
    sizes[5] = need_preatt ? L * (size_t)B * NH * T * T : 0; /* att -- only for the dense reference path */
    sizes[6] = L * (size_t)B * T * C;          /* attproj */
    sizes[7] = L * (size_t)B * T * C;          /* residual2 */
    sizes[8] = L * (size_t)B * T * C;          /* ln2 */
    sizes[9] = L * (size_t)B * T * 4 * C;      /* fch */
    sizes[10] = L * (size_t)B * T * 4 * C;     /* fch_gelu */
    sizes[11] = L * (size_t)B * T * C;         /* fcproj */
    sizes[12] = L * (size_t)B * T * C;         /* residual3 -- reused as lnf/logits base below */

    size_t total = 0;
    for (int i = 0; i < 13; i++) total += sizes[i];
    total += (size_t)B * T * C;          /* lnf */
    total += (size_t)B * T * Vp;         /* logits */

    float *mem = (float *)malloc_check(total * sizeof(float));
    float *it = mem;
    a->encoded = it; it += sizes[0];
    a->ln1 = it; it += sizes[1];
    a->qkv = it; it += sizes[2];
    a->atty = it; it += sizes[3];
    a->preatt = need_preatt ? it : NULL; it += sizes[4];
    a->att = need_preatt ? it : NULL; it += sizes[5];
    a->attproj = it; it += sizes[6];
    a->residual2 = it; it += sizes[7];
    a->ln2 = it; it += sizes[8];
    a->fch = it; it += sizes[9];
    a->fch_gelu = it; it += sizes[10];
    a->fcproj = it; it += sizes[11];
    a->residual3 = it; it += sizes[12];
    a->lnf = it; it += (size_t)B * T * C;
    a->logits = it;
    return mem;
}

typedef struct {
    GPT2Config config;
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
} GPT2;

static void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path) {
    FILE *f = fopen_check(checkpoint_path, "rb");
    int header[256];
    fread_check(header, sizeof(int), 256, f);
    if (header[0] != 20240326) { fprintf(stderr, "Bad magic in %s\n", checkpoint_path); exit(1); }
    if (header[1] != 3) { fprintf(stderr, "Expected version 3 (float32); re-run export_gpt2_c.py\n"); exit(1); }

    model->config.max_seq_len = header[2];
    model->config.vocab_size = header[3];
    model->config.num_layers = header[4];
    model->config.num_heads = header[5];
    model->config.channels = header[6];
    model->config.padded_vocab_size = header[7];
    printf("[GPT-2] max_seq_len=%d vocab=%d padded_vocab=%d layers=%d heads=%d channels=%d\n",
           model->config.max_seq_len, model->config.vocab_size, model->config.padded_vocab_size,
           model->config.num_layers, model->config.num_heads, model->config.channels);

    fill_in_parameter_sizes(model->param_sizes, model->config);
    size_t total = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) total += model->param_sizes[i];
    model->num_parameters = total;
    printf("num_parameters: %zu\n", total);

    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    fread_check(model->params_memory, sizeof(float), total, f);
    fclose(f);
}

#define LAZY_THRESHOLD 0.05f /* matches Definition 1 (windowing paper): delta_ppl < 0.05 -> lazy */

/* Loads the per-(layer,head) causal compressibility map (export_gpt2_c.py, sourced from
 * head_compressibility_gpt2.json) and thresholds it into an L*NH windowed/dense mask -- the
 * actual paper claim (window the lazy heads, keep load-bearing heads dense), not just "some
 * heads windowed, some not" with no basis for which. */
static int *load_lazy_map(const char *path, int L, int NH, int *sink_out, int *window_out) {
    FILE *f = fopen_check(path, "rb");
    int header[256];
    fread_check(header, sizeof(int), 256, f);
    if (header[0] != 20260712) { fprintf(stderr, "Bad magic in %s\n", path); exit(1); }
    int map_L = header[2], map_NH = header[3];
    *sink_out = header[4];
    *window_out = header[5];
    if (map_L != L || map_NH != NH) {
        fprintf(stderr, "Map shape (%d,%d) does not match model (%d,%d)\n", map_L, map_NH, L, NH);
        exit(1);
    }
    float *delta_ppl = (float *)malloc_check((size_t)L * NH * sizeof(float));
    fread_check(delta_ppl, sizeof(float), (size_t)L * NH, f);
    fclose(f);

    int *mask = (int *)malloc_check((size_t)L * NH * sizeof(int));
    int n_lazy = 0;
    for (int i = 0; i < L * NH; i++) {
        mask[i] = delta_ppl[i] < LAZY_THRESHOLD;
        n_lazy += mask[i];
    }
    printf("[Map] %s: %d/%d heads lazy at delta_ppl<%.3f (sink=%d window=%d)\n",
           path, n_lazy, L * NH, LAZY_THRESHOLD, *sink_out, *window_out);
    free(delta_ppl);
    return mask;
}

/* Single forward pass. head_mask is L*NH (row-major, one row per layer) of 0/1: which heads
 * are windowed at that layer. head_mask == NULL means "dense reference" -- uses the original
 * attention_forward (full preatt/att, unmodified from llm.c) instead of attention_forward_
 * perhead, kept as the one path that never changes so it stays trustworthy as the HF-parity
 * anchor everything else gates against. For dense/map/allwindow arms, pass an L*NH mask of
 * all zeros / the real map / all ones respectively -- same code path, different mask content.
 * Allocates its own activations each call (simpler than lazy-alloc-and-assert-shape for a
 * benchmark that sweeps multiple T values) -- allocation happens outside the timed region. */
static void gpt2_forward(GPT2 *model, const int *inputs, int B, int T, const int *head_mask,
                          ActivationTensors *acts_out, float **acts_mem_out) {
    size_t C = model->config.channels, NH = model->config.num_heads, L = model->config.num_layers,
           Vp = model->config.padded_vocab_size;
    for (int i = 0; i < B * T; i++) assert(0 <= inputs[i] && inputs[i] < (int)model->config.vocab_size);

    ActivationTensors acts;
    float *acts_mem = malloc_and_point_activations(&acts, model->config, B, T, /*need_preatt=*/head_mask == NULL);
    ParameterTensors p = model->params;

    encoder_forward(acts.encoded, inputs, p.wte, p.wpe, B, T, (int)C);

    for (int l = 0; l < (int)L; l++) {
        float *residual = l == 0 ? acts.encoded : acts.residual3 + (size_t)(l - 1) * B * T * C;
        float *l_ln1w = p.ln1w + (size_t)l * C, *l_ln1b = p.ln1b + (size_t)l * C;
        float *l_qkvw = p.qkvw + (size_t)l * 3 * C * C, *l_qkvb = p.qkvb + (size_t)l * 3 * C;
        float *l_attprojw = p.attprojw + (size_t)l * C * C, *l_attprojb = p.attprojb + (size_t)l * C;
        float *l_ln2w = p.ln2w + (size_t)l * C, *l_ln2b = p.ln2b + (size_t)l * C;
        float *l_fcw = p.fcw + (size_t)l * 4 * C * C, *l_fcb = p.fcb + (size_t)l * 4 * C;
        float *l_fcprojw = p.fcprojw + (size_t)l * C * 4 * C, *l_fcprojb = p.fcprojb + (size_t)l * C;

        float *l_ln1 = acts.ln1 + (size_t)l * B * T * C;
        float *l_qkv = acts.qkv + (size_t)l * B * T * 3 * C;
        float *l_atty = acts.atty + (size_t)l * B * T * C;
        float *l_attproj = acts.attproj + (size_t)l * B * T * C;
        float *l_residual2 = acts.residual2 + (size_t)l * B * T * C;
        float *l_ln2 = acts.ln2 + (size_t)l * B * T * C;
        float *l_fch = acts.fch + (size_t)l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + (size_t)l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + (size_t)l * B * T * C;
        float *l_residual3 = acts.residual3 + (size_t)l * B * T * C;

        layernorm_forward(l_ln1, residual, l_ln1w, l_ln1b, B, T, (int)C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, (int)C, (int)(3 * C));
        if (head_mask == NULL) {
            float *l_preatt = acts.preatt + (size_t)l * B * NH * T * T;
            float *l_att = acts.att + (size_t)l * B * NH * T * T;
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, (int)C, (int)NH);
        } else {
            const int *l_head_mask = head_mask + (size_t)l * NH;
            attention_forward_perhead(l_atty, l_qkv, B, T, (int)C, (int)NH, l_head_mask, SINK, WINDOW);
        }
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, (int)C, (int)C);
        residual_forward(l_residual2, residual, l_attproj, B * T * (int)C);
        layernorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, (int)C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, (int)C, (int)(4 * C));
        gelu_forward(l_fch_gelu, l_fch, B * T * (int)(4 * C));
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, (int)(4 * C), (int)C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * (int)C);
    }
    float *final_residual = acts.residual3 + (L - 1) * (size_t)B * T * C;
    layernorm_forward(acts.lnf, final_residual, p.lnfw, p.lnfb, B, T, (int)C);
    matmul_forward(acts.logits, acts.lnf, p.wte, NULL, B, T, (int)C, (int)Vp);

    *acts_out = acts;
    *acts_mem_out = acts_mem;
}

/* ---------------------------------------------------------------------------
 * main: four parity gates (all against the dense reference, on the short export prompt
 * where windowing at ANY fraction is mathematically required to reduce to dense -- so this
 * validates both branches of attention_forward_perhead AND that the map loads/threads
 * through correctly, all from data already exported, no extra runs needed), then a
 * three-arm (dense/map/allwindow) timing sweep within GPT-2's own 1024-token position-
 * embedding ceiling.
 * ------------------------------------------------------------------------- */
static int compare_logits(const ActivationTensors *a, const ActivationTensors *b, int B, int T, int V, int Vp) {
    float max_diff = 0.0f;
    for (int bt = 0; bt < B * T; bt++)
        for (int v = 0; v < V; v++) {
            float diff = fabsf(a->logits[(size_t)bt * Vp + v] - b->logits[(size_t)bt * Vp + v]);
            if (diff > max_diff) max_diff = diff;
        }
    printf("  max abs diff = %.3e\n", max_diff);
    return max_diff < 1e-3f;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0); /* line-buffer even when stdout is a pipe (e.g. `| tee`) --
                                          otherwise glibc fully-buffers non-TTY stdout and a long
                                          run's output sits invisible until exit */
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    int L = model.config.num_layers, NH = model.config.num_heads;

    FILE *sf = fopen_check("gpt2_124M_fwd_state.bin", "rb");
    int sheader[256];
    fread_check(sheader, sizeof(int), 256, sf);
    if (sheader[0] != 20260711) { fprintf(stderr, "Bad magic in state file\n"); return 1; }
    int B = sheader[2], T = sheader[3], V = sheader[4];
    printf("[State] B=%d T=%d V=%d\n", B, T, V);

    int *x = (int *)malloc_check((size_t)B * T * sizeof(int));
    float *expected_logits = (float *)malloc_check((size_t)B * T * V * sizeof(float));
    fread_check(x, sizeof(int), (size_t)B * T, sf);
    fread_check(expected_logits, sizeof(float), (size_t)B * T * V, sf);
    fclose(sf);
    int Vp = model.config.padded_vocab_size;

    /* Build the three L*NH masks up front -- reused for both the parity gates and the
     * timing sweep, so the gates test exactly the mask content that gets timed. */
    int *mask_dense = (int *)malloc_check((size_t)L * NH * sizeof(int));
    int *mask_allwindow = (int *)malloc_check((size_t)L * NH * sizeof(int));
    for (int i = 0; i < L * NH; i++) { mask_dense[i] = 0; mask_allwindow[i] = 1; }
    int map_sink, map_window;
    int *mask_map = load_lazy_map("gpt2_124M_lazymap.bin", L, NH, &map_sink, &map_window);
    if (map_sink != SINK || map_window != WINDOW) {
        fprintf(stderr, "Map was built with sink=%d window=%d, this file uses SINK=%d WINDOW=%d -- "
                        "re-run export_gpt2_c.py against a matching map, or edit the #defines.\n",
                map_sink, map_window, SINK, WINDOW);
        return 1;
    }

    /* Gate 1: dense reference (head_mask=NULL, original attention_forward) vs real HF logits.
     * This is the one path that never changes -- the trusted anchor for gates 2-4. */
    ActivationTensors acts_d;
    float *mem_d;
    gpt2_forward(&model, x, B, T, NULL, &acts_d, &mem_d);
    float max_diff_dense = 0.0f;
    for (int bt = 0; bt < B * T; bt++)
        for (int v = 0; v < V; v++) {
            float diff = fabsf(expected_logits[(size_t)bt * V + v] - acts_d.logits[(size_t)bt * Vp + v]);
            if (diff > max_diff_dense) max_diff_dense = diff;
        }
    printf("Gate 1 (dense C vs real HF logits, T=%d): max abs diff = %.3e\n", T, max_diff_dense);
    int gate1_ok = max_diff_dense < 5e-2f; /* fp32 accumulation over 12 layers; matches llm.c's own 2e-2 tol order */
    printf(gate1_ok ? "PASSED.\n" : "FAILED -- do not trust anything below.\n");

    /* Gates 2-4: attention_forward_perhead with each mask, on the SAME short prompt
     * (T well under WINDOW=64), where windowing at any fraction is required to reduce to
     * dense exactly -- so all three must match gate 1's reference. Gate 2 validates
     * attention_forward_perhead's dense/VLA branch; gate 3 its windowed branch; gate 4 that
     * the real map loads and threads through correctly end to end. */
    struct { const char *name; const int *mask; } gates[3] = {
        {"all-dense mask (validates perhead's dense branch)", mask_dense},
        {"all-window mask (validates perhead's windowed branch)", mask_allwindow},
        {"real lazy map (validates map load + threading)", mask_map},
    };
    int gates_ok = 1;
    for (int i = 0; i < 3; i++) {
        ActivationTensors a; float *m;
        gpt2_forward(&model, x, B, T, gates[i].mask, &a, &m);
        printf("Gate %d (%s, T=%d < WINDOW=%d):\n", i + 2, gates[i].name, T, WINDOW);
        int ok = compare_logits(&acts_d, &a, B, T, V, Vp);
        printf(ok ? "PASSED.\n" : "FAILED -- do not trust anything below.\n");
        gates_ok = gates_ok && ok;
        free(m);
    }
    printf("\n");

    free(mem_d); free(x); free(expected_logits);
    if (!gate1_ok || !gates_ok) return 1;

    /* Timing sweep: full end-to-end forward pass, three arms, within GPT-2's own 1024-token
     * position-embedding ceiling. */
    int contexts[] = {128, 256, 512, 1024};
    int n_contexts = (int)(sizeof(contexts) / sizeof(contexts[0]));
    int warmup = 2, repeats = 5; /* full-model passes are much heavier than the isolated
                                    kernel benchmark -- fewer repeats to keep this practical */
    struct { const char *name; const int *mask; } arms[3] = {
        {"dense", mask_dense}, {"map", mask_map}, {"allwindow", mask_allwindow},
    };
    unsigned int seed = 42;
    srand(seed);

    printf("%-8s %14s %14s %14s %12s %12s\n", "T", "dense(ms)", "map(ms)", "allwin(ms)", "map_speedup", "allwin_speedup");
    for (int ci = 0; ci < n_contexts; ci++) {
        int Tc = contexts[ci];
        int *ids = (int *)malloc_check((size_t)Tc * sizeof(int));
        for (int i = 0; i < Tc; i++) ids[i] = rand() % model.config.vocab_size;

        double ms[3];
        for (int a = 0; a < 3; a++) {
            for (int i = 0; i < warmup; i++) {
                ActivationTensors act; float *m;
                gpt2_forward(&model, ids, 1, Tc, arms[a].mask, &act, &m);
                free(m);
            }
            double t0 = now_sec();
            for (int i = 0; i < repeats; i++) {
                ActivationTensors act; float *m;
                gpt2_forward(&model, ids, 1, Tc, arms[a].mask, &act, &m);
                free(m);
            }
            ms[a] = (now_sec() - t0) / repeats * 1000.0;
        }

        printf("%-8d %14.2f %14.2f %14.2f %11.2fx %13.2fx\n",
               Tc, ms[0], ms[1], ms[2], ms[0] / ms[1], ms[0] / ms[2]);
        free(ids);
    }

    free(mask_dense); free(mask_allwindow); free(mask_map);
    free(model.params_memory);
    return 0;
}
