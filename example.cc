#include "otsu.cuh"
#include <iostream>

int32_t main() {
    uint32_t n_bins = 256;
    uint32_t *cpu_histogram = new uint32_t[n_bins];
    uint32_t cpu_otsu_threshold;
    uint32_t *gpu_histogram;
    uint32_t *gpu_otsu_threshold;

    srand(42);
    for(uint32_t i = 0; i < n_bins; ++i) {
        cpu_histogram[i] = rand() % 255;
    }

    cudaMalloc((void **) &gpu_histogram, n_bins * sizeof(*gpu_histogram));
    cudaMalloc((void **) &gpu_otsu_threshold, sizeof(*gpu_otsu_threshold));
    cudaMemcpy(gpu_histogram, cpu_histogram, n_bins * sizeof(*gpu_histogram), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    Otsu otsu = Otsu(n_bins, stream);
    otsu.run_async(gpu_histogram, gpu_otsu_threshold, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaAssert(cudaMemcpy(&cpu_otsu_threshold, gpu_otsu_threshold, sizeof(*gpu_otsu_threshold), cudaMemcpyDeviceToHost)
    );

    printf("Otsu threshold: %u\n", cpu_otsu_threshold);

    cudaFree(gpu_histogram);
    cudaFree(gpu_otsu_threshold);
    delete[] cpu_histogram;
}
