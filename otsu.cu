#include "otsu.cuh"
#include <iostream>
#include <stdexcept>

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32


inline uint32_t calc_blocks(uint32_t i, uint32_t size) {
    return i / size + (i % size != 0);
}


void otsu_async(
    uint32_t *otsu_threshold, uint32_t *histogram, uint32_t *threshold_sums, uint64_t *sums, longlong2 *variance,
    uint32_t n_bins, uint32_t n_thresholds, dim3 grid_all, dim3 block_all, dim3 grid_score, dim3 block_score,
    cudaStream_t stream
);


Otsu::Otsu(uint32_t n_bins, cudaStream_t stream) {
    this->n_bins = n_bins;
    this->n_thresholds = n_bins;

    if(this->n_bins % 32 != 0) {
        throw std::runtime_error("n_bins must be a multiple of 32");
    }

    this->block_all = dim3(this->n_bins, 1, 1);
    this->grid_all =
        dim3(calc_blocks(this->n_bins, this->block_all.x), calc_blocks(this->n_thresholds, this->block_all.y), 1);
    this->block_score = dim3(this->n_thresholds, 1, 1);
    this->grid_score = dim3(calc_blocks(this->n_thresholds, this->block_score.x), 1, 1);

    cudaAssert(cudaMalloc((void **) &this->gpu_threshold_sums, this->n_bins * sizeof(*this->gpu_threshold_sums)));
    cudaAssert(cudaMalloc((void **) &this->gpu_sums, this->n_bins * sizeof(*this->gpu_sums)));
    cudaAssert(cudaMalloc((void **) &this->gpu_variances, this->n_bins * sizeof(*this->gpu_variances)));
}

Otsu::~Otsu() {
    if(this->gpu_threshold_sums) {
        cudaAssert(cudaFree(this->gpu_threshold_sums));
        this->gpu_threshold_sums = nullptr;
    }
    if(this->gpu_sums) {
        cudaAssert(cudaFree(this->gpu_sums));
        this->gpu_sums = nullptr;
    }
    if(this->gpu_variances) {
        cudaAssert(cudaFree(this->gpu_variances));
        this->gpu_variances = nullptr;
    }
}

void Otsu::run_async(uint32_t *gpu_histogram, uint32_t *otsu_threshold, cudaStream_t stream) {
    otsu_async(
        otsu_threshold, gpu_histogram, this->gpu_threshold_sums, this->gpu_sums, this->gpu_variances, this->n_bins,
        this->n_thresholds, this->grid_all, this->block_all, this->grid_score, this->block_score, stream
    );
}


__device__ __forceinline__ uint32_t lane_id(uint32_t tid) {
    return tid & (32 - 1);
}

__device__ __forceinline__ uint32_t warp_id(uint32_t tid) {
    return tid / 32;
}


template <typename T> __device__ T warp_sum(T value, uint32_t mask) {
    // Shuffle values from the warp to the first thread in the warp
#pragma unroll
    for(uint32_t i = WARP_SIZE / 2; 0 < i; i /= 2) {
        T value_tmp = __shfl_down_sync(mask, value, i);
        value += value_tmp;
    }

    return value;
}

template <bool broadcast, typename T>
__device__ T block_sum(T value, uint32_t tid, uint32_t n_threads, T shared_memory[]) {
    // Leverage the fast interconnect to reduce warps first
    value = warp_sum(value, FULL_MASK);

    // If this thread is #0 for the warp, make it save to shared
    if(lane_id(tid) == 0) {
        shared_memory[tid / WARP_SIZE] = value;
    }

    __syncthreads();

    // Take all the intermediate values from the warps and collect them in a single warp. Reduce this warp using shuffle
    uint32_t n_warps = n_threads / WARP_SIZE;

    if(tid < n_warps) {
        value = shared_memory[tid];

        // Create a mask for the active threads
        uint32_t mask = (1 << n_warps) - 1;

#pragma unroll
        for(uint32_t i = n_warps / 2; 0 < i; i /= 2) {
            T value_tmp = __shfl_down_sync(mask, value, i);
            value += value_tmp;
        }
    }

    // If broadcasting: Make sure all threads have the value; otherwise, only tid=0 will have the correct value
    if(broadcast == true) {
        if(tid == 0) {
            shared_memory[0] = value;
        }

        __syncthreads();

        value = shared_memory[0];
    }

    return value;
}

template <typename T> __device__ T warp_min(T value, uint32_t mask) {
    // Shuffle values from the warp to the first thread in the warp
#pragma unroll
    for(uint32_t i = WARP_SIZE / 2; 0 < i; i /= 2) {
        T value_tmp = __shfl_down_sync(mask, value, i);
        value = min(value, value_tmp);
    }

    return value;
}

template <bool broadcast, typename T>
__device__ T block_min(T value, uint32_t tid, uint32_t n_threads, T shared_memory[]) {
    // Leverage the fast interconnect to reduce warps first
    value = warp_min(value, FULL_MASK);

    // If this thread is #0 for the warp, make it save to shared
    if(lane_id(tid) == 0) {
        shared_memory[tid / WARP_SIZE] = value;
    }

    __syncthreads();

    // Take all the intermediate values from the warps and collect them in a single warp. Reduce this warp using shuffle
    uint32_t n_warps = n_threads / WARP_SIZE;

    if(tid < n_warps) {
        value = shared_memory[tid];

        // Create a mask for the active threads
        uint32_t mask = (1 << n_warps) - 1;

#pragma unroll
        for(uint32_t i = n_warps / 2; 0 < i; i /= 2) {
            T value_tmp = __shfl_down_sync(mask, value, i);
            value = min(value, value_tmp);
        }
    }

    // If broadcasting: Make sure all threads have the value; otherwise, only tid=0 will have the correct value
    if(broadcast == true) {
        if(tid == 0) {
            shared_memory[0] = value;
        }

        __syncthreads();

        value = shared_memory[0];
    }

    return value;
}


__global__ void otsu_mean(uint32_t *histogram, uint32_t *threshold_sums, uint64_t *sums, uint32_t n_bins) {
    extern __shared__ uint64_t shared_memory_u64[];

    uint32_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threshold = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t threshold_sum_above = 0;
    uint64_t sum_above = 0;

    if(bin_idx >= threshold) {
        uint32_t value = histogram[bin_idx];
        threshold_sum_above = value;
        sum_above = value * bin_idx;
    }

    threshold_sum_above = block_sum<false>(threshold_sum_above, bin_idx, n_bins, (uint32_t *) shared_memory_u64);
    sum_above = block_sum<false>(sum_above, bin_idx, n_bins, (uint64_t *) shared_memory_u64);

    if(bin_idx == 0) {
        threshold_sums[threshold] = threshold_sum_above;
        sums[threshold] = sum_above;
    }
}

__global__ void
otsu_variance(longlong2 *variance, uint32_t *histogram, uint32_t *threshold_sums, uint64_t *sums, uint32_t n_bins) {
    extern __shared__ int64_t shared_memory_i64[];

    uint32_t bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t threshold = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t n_samples = threshold_sums[0];
    uint32_t n_samples_above = threshold_sums[threshold];
    uint32_t n_samples_below = n_samples - n_samples_above;

    uint64_t total_sum = sums[0];
    uint64_t sum_above = sums[threshold];
    uint64_t sum_below = total_sum - sum_above;

    int32_t threshold_variance_above = 0;
    int32_t threshold_variance_below = 0;
    if(bin_idx >= threshold) {
        uint32_t mean = sum_above / n_samples_above;
        int32_t sigma = bin_idx - mean;
        threshold_variance_above = sigma * sigma;
    } else {
        uint32_t mean = sum_below / n_samples_below;
        int32_t sigma = bin_idx - mean;
        threshold_variance_below = sigma * sigma;
    }

    uint32_t bin_count = histogram[bin_idx];
    int64_t threshold_variance_above64 = threshold_variance_above * bin_count;
    int64_t threshold_variance_below64 = threshold_variance_below * bin_count;
    threshold_variance_above64 = block_sum<false>(threshold_variance_above64, bin_idx, n_bins, shared_memory_i64);
    threshold_variance_below64 = block_sum<false>(threshold_variance_below64, bin_idx, n_bins, shared_memory_i64);

    if(bin_idx == 0) {
        variance[threshold] = make_longlong2(threshold_variance_above64, threshold_variance_below64);
    }
}

__global__ void
otsu_score(uint32_t *otsu_threshold, uint32_t *threshold_sums, longlong2 *variance, uint32_t n_thresholds) {
    extern __shared__ float shared_memory_f32[];

    uint32_t threshold = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t n_samples = threshold_sums[0];
    uint32_t n_samples_above = threshold_sums[threshold];
    uint32_t n_samples_below = n_samples - n_samples_above;

    float threshold_mean_above = (float) n_samples_above / n_samples;
    float threshold_mean_below = (float) n_samples_below / n_samples;

    longlong2 variances = variance[threshold];
    float variance_above = (float) variances.x / n_samples_above;
    float variance_below = (float) variances.y / n_samples_below;

    float above = threshold_mean_above * variance_above;
    float below = threshold_mean_below * variance_below;
    float score = above + below;

    float original_score = score;

    score = block_min<true>(score, threshold, n_thresholds, shared_memory_f32);

    // We found the minimum score, but we need to find the threshold. If we find the thread with the minimum score, we
    // know which threshold it is
    if(original_score == score) {
        *otsu_threshold = threshold - 1;
    }
}


void otsu_async(
    uint32_t *otsu_threshold, uint32_t *histogram, uint32_t *threshold_sums, uint64_t *sums, longlong2 *variance,
    uint32_t n_bins, uint32_t n_thresholds, dim3 grid_all, dim3 block_all, dim3 grid_score, dim3 block_score,
    cudaStream_t stream
) {
    uint32_t shared_memory;

    shared_memory = n_bins / WARP_SIZE * sizeof(uint64_t);
    otsu_mean<<<grid_all, block_all, shared_memory, stream>>>(histogram, threshold_sums, sums, n_bins);
    otsu_variance<<<grid_all, block_all, shared_memory, stream>>>(variance, histogram, threshold_sums, sums, n_bins);

    shared_memory = n_bins / WARP_SIZE * sizeof(float);
    otsu_score<<<grid_score, block_score, shared_memory, stream>>>(
        otsu_threshold, threshold_sums, variance, n_thresholds
    );
}
