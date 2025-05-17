#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>


#define cudaAssert(ans)                                                                                                \
    {                                                                                                                  \
        cuda_error_handler((ans), __FILE__, __LINE__);                                                                 \
    }

inline void cuda_error_handler(cudaError_t code, const char *file, int line, bool abort = true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "CUDA error with message \"%s\" in %s on line %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

class Otsu {
    private:
    uint32_t n_bins;
    uint32_t n_thresholds;

    dim3 block_all;
    dim3 grid_all;
    dim3 block_score;
    dim3 grid_score;

    uint32_t *gpu_threshold_sums = nullptr;
    uint64_t *gpu_sums = nullptr;
    longlong2 *gpu_variances = nullptr;

    public:
    void run_async(uint32_t *gpu_histogram, uint32_t *otsu_threshold, cudaStream_t stream);
    Otsu(uint32_t n_bins, cudaStream_t stream);
    ~Otsu();
};
