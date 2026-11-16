// GPU audio denoiser: STFT -> spectral subtraction -> inverse STFT
// Three implementations of the same pipeline: CPU (1 thread), OpenMP, CUDA.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

struct Cpx { float re, im; };

struct Params {
    int N;            // FFT size (power of two)
    int logN;
    int hop;          // hop size (N/4 -> 75% overlap)
    float alpha;      // over-subtraction factor
    float floorGain;  // minimum gain per bin
    float noiseSec;   // leading part of the file assumed to be noise only
};

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__,          \
                    cudaGetErrorString(e_));                                  \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static double nowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

__host__ __device__ inline unsigned bitrev(unsigned x, int bits) {
    unsigned r = 0;
    for (int i = 0; i < bits; i++) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

// ------------------------------------------------------------------ WAV I/O
struct Wav { int sampleRate = 0; std::vector<float> data; };   // mono, [-1, 1]

static bool readWav(const char* path, Wav& w) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    char id[4]; uint32_t sz = 0;
    if (fread(id, 1, 4, f) != 4 || memcmp(id, "RIFF", 4) != 0 ||
        fread(&sz, 4, 1, f) != 1 ||
        fread(id, 1, 4, f) != 4 || memcmp(id, "WAVE", 4) != 0) {
        fprintf(stderr, "Not a RIFF/WAVE file\n"); fclose(f); return false;
    }
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t rate = 0; bool haveFmt = false;
    while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (memcmp(id, "fmt ", 4) == 0) {
            std::vector<uint8_t> buf(sz);
            if (sz < 16 || fread(buf.data(), 1, sz, f) != sz) break;
            memcpy(&fmt, &buf[0], 2);  memcpy(&ch, &buf[2], 2);
            memcpy(&rate, &buf[4], 4); memcpy(&bits, &buf[14], 2);
            haveFmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            if (!haveFmt || fmt != 1 || bits != 16 || ch < 1) {
                fprintf(stderr, "Only 16-bit PCM WAV is supported\n");
                fclose(f); return false;
            }
            size_t total = sz / 2;
            std::vector<int16_t> raw(total);
            size_t got = fread(raw.data(), 2, total, f);
            size_t n = got / ch;
            w.data.resize(n);
            for (size_t i = 0; i < n; i++) {
                float s = 0;
                for (int c = 0; c < ch; c++) s += raw[i * ch + c];
                w.data[i] = s / (32768.0f * ch);          // mix down to mono
            }
            w.sampleRate = (int)rate;
            fclose(f);
            return true;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
        if (sz & 1) fseek(f, 1, SEEK_CUR);                // chunk padding byte
    }
    fprintf(stderr, "No usable data chunk found\n");
    fclose(f);
    return false;
}

static bool writeWav(const char* path, const float* x, size_t n, int sampleRate) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    uint32_t dataBytes = (uint32_t)(n * 2), riff = 36 + dataBytes, fmtSize = 16;
    uint16_t fmt = 1, ch = 1, bits = 16, align = 2;
    uint32_t rate = (uint32_t)sampleRate, byteRate = rate * 2;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fmtSize, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f);
    fwrite(&byteRate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dataBytes, 4, 1, f);
    std::vector<int16_t> out(n);
    for (size_t i = 0; i < n; i++) {
        float v = std::max(-1.0f, std::min(1.0f, x[i]));
        out[i] = (int16_t)lrintf(v * 32767.0f);
    }
    fwrite(out.data(), 2, n, f);
    fclose(f);
    return true;
}

// ---------------------------------------------------------------- CPU path
// In-place radix-2 FFT. Input must already be in bit-reversed order.
// tw[k] = exp(-2*pi*i*k/N), k < N/2. The inverse uses the conjugate (no 1/N).
static void fftCore(Cpx* a, int N, const Cpx* tw, bool inverse) {
    for (int len = 2; len <= N; len <<= 1) {
        int half = len >> 1, step = N / len;
        for (int i = 0; i < N; i += len) {
            for (int k = 0; k < half; k++) {
                Cpx w = tw[k * step];
                if (inverse) w.im = -w.im;
                Cpx u = a[i + k], v = a[i + k + half];
                float tr = v.re * w.re - v.im * w.im;
                float ti = v.re * w.im + v.im * w.re;
                a[i + k].re = u.re + tr;        a[i + k].im = u.im + ti;
                a[i + k + half].re = u.re - tr; a[i + k + half].im = u.im - ti;
            }
        }
    }
}

static void denoiseCPU(const std::vector<float>& x, int F, int K, const Params& P,
                       const std::vector<float>& win, const std::vector<Cpx>& tw,
                       std::vector<float>& out, int threads) {
    const int N = P.N, hop = P.hop, logN = P.logN;
    const int Lp = (int)x.size();
    std::vector<Cpx> A((size_t)F * N), B((size_t)F * N);
    std::vector<float> noisePow(N);
    omp_set_num_threads(threads);

    // 1) window + forward FFT of every frame
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < F; f++) {
        Cpx* a = &A[(size_t)f * N];
        const float* src = &x[(size_t)f * hop];
        for (int i = 0; i < N; i++) {
            unsigned r = bitrev((unsigned)i, logN);
            a[r].re = src[i] * win[i];
            a[r].im = 0.0f;
        }
        fftCore(a, N, tw.data(), false);
    }

    // 2) noise power per bin = mean over the first K frames
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < N; b++) {
        float s = 0.0f;
        for (int f = 0; f < K; f++) {
            Cpx c = A[(size_t)f * N + b];
            s += c.re * c.re + c.im * c.im;
        }
        noisePow[b] = s / K;
    }

    // 3) spectral subtraction + inverse FFT
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < F; f++) {
        const Cpx* a = &A[(size_t)f * N];
        Cpx* o = &B[(size_t)f * N];
        for (int b = 0; b < N; b++) {
            Cpx c = a[b];
            float p = c.re * c.re + c.im * c.im;
            float g = (p - P.alpha * noisePow[b]) / (p + 1e-12f);
            if (g < P.floorGain) g = P.floorGain;
            unsigned r = bitrev((unsigned)b, logN);
            o[r].re = c.re * g;
            o[r].im = c.im * g;
        }
        fftCore(o, N, tw.data(), true);
    }

    // 4) weighted overlap-add, written as a gather (one output sample at a time)
    out.assign(Lp, 0.0f);
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < Lp; n++) {
        int fmax = n / hop; if (fmax > F - 1) fmax = F - 1;
        int t = n - N + 1;
        int fmin = (t <= 0) ? 0 : (t + hop - 1) / hop;
        float acc = 0.0f, ws = 0.0f;
        for (int f = fmin; f <= fmax; f++) {
            int i = n - f * hop;
            float w = win[i];
            acc += B[(size_t)f * N + i].re * w;
            ws += w * w;
        }
        out[n] = (ws > 1e-3f) ? acc / (ws * N) : 0.0f;
    }
}

// --------------------------------------------------------------- CUDA path
static const int TPB = 256;   // threads per block
static inline unsigned gridFor(size_t total) {
    return (unsigned)((total + TPB - 1) / TPB);
}

// one thread per (frame, sample): apply window, store in bit-reversed position
__global__ void kWindowBitrev(const float* x, const float* win, Cpx* spec,
                              int N, int logN, int hop, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t f = idx / N;
    int i = (int)(idx % N);
    unsigned r = bitrev((unsigned)i, logN);
    Cpx c;
    c.re = x[f * hop + i] * win[i];
    c.im = 0.0f;
    spec[f * N + r] = c;
}

// one FFT stage: one thread per butterfly, for all frames at once.
// Every butterfly touches its own pair of elements, so there is no data race.
__global__ void kButterfly(Cpx* a, const Cpx* tw, int N, int len,
                           size_t total, int inverse) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int halfN = N >> 1;
    size_t f = idx / halfN;
    int j = (int)(idx % halfN);
    int half = len >> 1;
    int group = j / half, k = j % half;
    size_t i0 = f * N + (size_t)group * len + k;
    size_t i1 = i0 + half;
    Cpx w = tw[k * (N / len)];
    if (inverse) w.im = -w.im;
    Cpx u = a[i0], v = a[i1];
    float tr = v.re * w.re - v.im * w.im;
    float ti = v.re * w.im + v.im * w.re;
    Cpx p, q;
    p.re = u.re + tr; p.im = u.im + ti;
    q.re = u.re - tr; q.im = u.im - ti;
    a[i0] = p;
    a[i1] = q;
}

// one thread per frequency bin
__global__ void kNoise(const Cpx* spec, float* noisePow, int N, int K) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= N) return;
    float s = 0.0f;
    for (int f = 0; f < K; f++) {
        Cpx c = spec[(size_t)f * N + b];
        s += c.re * c.re + c.im * c.im;
    }
    noisePow[b] = s / K;
}

// one thread per (frame, bin): apply gain, store bit-reversed for the inverse FFT
__global__ void kGainBitrev(const Cpx* spec, const float* noisePow, Cpx* out,
                            int N, int logN, size_t total,
                            float alpha, float floorGain) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t f = idx / N;
    int b = (int)(idx % N);
    Cpx c = spec[idx];
    float p = c.re * c.re + c.im * c.im;
    float g = (p - alpha * noisePow[b]) / (p + 1e-12f);
    if (g < floorGain) g = floorGain;
    unsigned r = bitrev((unsigned)b, logN);
    Cpx o;
    o.re = c.re * g;
    o.im = c.im * g;
    out[f * N + r] = o;
}

// one thread per output sample: gather from the (at most N/hop) covering frames
__global__ void kOverlapAdd(const Cpx* frames, const float* win, float* out,
                            int N, int hop, int F, int Lp) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= Lp) return;
    int fmax = n / hop; if (fmax > F - 1) fmax = F - 1;
    int t = n - N + 1;
    int fmin = (t <= 0) ? 0 : (t + hop - 1) / hop;
    float acc = 0.0f, ws = 0.0f;
    for (int f = fmin; f <= fmax; f++) {
        int i = n - f * hop;
        float w = win[i];
        acc += frames[(size_t)f * N + i].re * w;
        ws += w * w;
    }
    out[n] = (ws > 1e-3f) ? acc / (ws * N) : 0.0f;
}

static void denoiseGPU(const std::vector<float>& x, int F, int K, const Params& P,
                       const std::vector<float>& win, const std::vector<Cpx>& tw,
                       std::vector<float>& out, double& msTotal, double& msCompute) {
    const int N = P.N, hop = P.hop, logN = P.logN;
    const int Lp = (int)x.size();
    const size_t FN = (size_t)F * N;

    float *dX = nullptr, *dWin = nullptr, *dNoise = nullptr, *dOut = nullptr;
    Cpx *dTw = nullptr, *dA = nullptr, *dB = nullptr;

    double t0 = nowMs();
    CUDA_CHECK(cudaMalloc((void**)&dX, (size_t)Lp * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dOut, (size_t)Lp * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dWin, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dNoise, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dTw, (size_t)(N / 2) * sizeof(Cpx)));
    CUDA_CHECK(cudaMalloc((void**)&dA, FN * sizeof(Cpx)));
    CUDA_CHECK(cudaMalloc((void**)&dB, FN * sizeof(Cpx)));
    CUDA_CHECK(cudaMemcpy(dX, x.data(), (size_t)Lp * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWin, win.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTw, tw.data(), (size_t)(N / 2) * sizeof(Cpx), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = nowMs();

    kWindowBitrev<<<gridFor(FN), TPB>>>(dX, dWin, dA, N, logN, hop, FN);
    for (int len = 2; len <= N; len <<= 1)
        kButterfly<<<gridFor(FN / 2), TPB>>>(dA, dTw, N, len, FN / 2, 0);
    kNoise<<<gridFor((size_t)N), TPB>>>(dA, dNoise, N, K);
    kGainBitrev<<<gridFor(FN), TPB>>>(dA, dNoise, dB, N, logN, FN, P.alpha, P.floorGain);
    for (int len = 2; len <= N; len <<= 1)
        kButterfly<<<gridFor(FN / 2), TPB>>>(dB, dTw, N, len, FN / 2, 1);
    kOverlapAdd<<<gridFor((size_t)Lp), TPB>>>(dB, dWin, dOut, N, hop, F, Lp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t2 = nowMs();

    out.resize(Lp);
    CUDA_CHECK(cudaMemcpy(out.data(), dOut, (size_t)Lp * sizeof(float), cudaMemcpyDeviceToHost));
    double t3 = nowMs();

    cudaFree(dX); cudaFree(dOut); cudaFree(dWin); cudaFree(dNoise);
    cudaFree(dTw); cudaFree(dA); cudaFree(dB);

    msTotal = t3 - t0;      // allocation + transfers + compute
    msCompute = t2 - t1;    // kernels only
}

static float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s input.wav output.wav [alpha=2.0] [floor=0.05] [noise_sec=0.5]\n", argv[0]);
        return 1;
    }
    Params P;
    P.N = 2048; P.logN = 11; P.hop = P.N / 4;
    P.alpha     = (argc > 3) ? (float)atof(argv[3]) : 2.0f;
    P.floorGain = (argc > 4) ? (float)atof(argv[4]) : 0.05f;
    P.noiseSec  = (argc > 5) ? (float)atof(argv[5]) : 0.5f;

    Wav in;
    if (!readWav(argv[1], in)) return 1;
    const size_t L = in.data.size();
    if (L == 0) { fprintf(stderr, "Empty audio\n"); return 1; }

    // pad so that the signal is covered by a whole number of frames
    const int N = P.N, hop = P.hop;
    size_t Fsz = (L <= (size_t)N) ? 1 : 1 + (L - N + hop - 1) / hop;
    if (Fsz * (size_t)N > ((size_t)1 << 28)) {
        fprintf(stderr, "Audio too long for this demo (limit is about 25 min at 44.1 kHz)\n");
        return 1;
    }
    const int F = (int)Fsz;
    const size_t Lp = (size_t)(F - 1) * hop + N;
    std::vector<float> x(Lp, 0.0f);
    std::copy(in.data.begin(), in.data.end(), x.begin());

    // frames that lie completely inside the leading noise-only part
    long long noiseSamples = (long long)(P.noiseSec * in.sampleRate);
    int K = (noiseSamples >= N) ? (int)((noiseSamples - N) / hop + 1) : 1;
    if (K > F) K = F;

    const double PI = 3.14159265358979323846;
    std::vector<float> win(N);
    for (int i = 0; i < N; i++) win[i] = (float)(0.5 - 0.5 * cos(2.0 * PI * i / N));
    std::vector<Cpx> tw(N / 2);
    for (int k = 0; k < N / 2; k++) {
        tw[k].re = (float)cos(-2.0 * PI * k / N);
        tw[k].im = (float)sin(-2.0 * PI * k / N);
    }

    printf("Input: %s | %d Hz | %.2f s | %zu samples\n",
           argv[1], in.sampleRate, (double)L / in.sampleRate, L);
    printf("FFT size %d, hop %d, frames %d, noise frames %d\n\n", N, hop, F, K);

    // warm up the GPU so context creation is not counted in the timings
    { void* p = nullptr; CUDA_CHECK(cudaMalloc(&p, 1 << 20)); CUDA_CHECK(cudaFree(p)); }

    int maxThreads = omp_get_max_threads();
    std::vector<float> outCpu, outOmp, outGpu;

    double a0 = nowMs();
    denoiseCPU(x, F, K, P, win, tw, outCpu, 1);
    double msCpu = nowMs() - a0;

    double b0 = nowMs();
    denoiseCPU(x, F, K, P, win, tw, outOmp, maxThreads);
    double msOmp = nowMs() - b0;

    double msGpuTotal = 0, msGpuCompute = 0;
    denoiseGPU(x, F, K, P, win, tw, outGpu, msGpuTotal, msGpuCompute);

    printf("%-28s %12s %10s\n", "Version", "Time (ms)", "Speedup");
    printf("%-28s %12.1f %10.2f\n", "CPU, 1 thread", msCpu, 1.0);
    char label[64];
    snprintf(label, sizeof(label), "OpenMP, %d threads", maxThreads);
    printf("%-28s %12.1f %10.2f\n", label, msOmp, msCpu / msOmp);
    printf("%-28s %12.1f %10.2f\n", "CUDA, with transfers", msGpuTotal, msCpu / msGpuTotal);
    printf("%-28s %12.1f %10.2f\n", "CUDA, kernels only", msGpuCompute, msCpu / msGpuCompute);

    printf("\nMax |CPU - OpenMP| = %.3e\n", maxAbsDiff(outCpu, outOmp));
    printf("Max |CPU - CUDA|   = %.3e\n", maxAbsDiff(outCpu, outGpu));

    if (!writeWav(argv[2], outGpu.data(), L, in.sampleRate)) {
        fprintf(stderr, "Cannot write %s\n", argv[2]);
        return 1;
    }
    printf("\nWrote %s (CUDA result)\n", argv[2]);
    return 0;
}
