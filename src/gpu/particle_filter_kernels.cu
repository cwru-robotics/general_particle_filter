/*
 * Copyright (C) 2017 Russell Jackson & CWRU Robotics
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "general_particle_filter/gpu/particle_filter.h"

namespace gpu_pf
{

//All three kernels run 512 threads per workgroup
//Must be a power of two


////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing
extern const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern const uint MIN_SHORT_ARRAY_SIZE = 4;
extern const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;


////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline __device__ double scan1Inclusive(double idata, volatile double *s_Data, uint size)
{
  uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
  s_Data[pos] = 0;
  pos += size;
  s_Data[pos] = idata;

  for (uint offset = 1; offset < size; offset <<= 1)
  {
    __syncthreads();
    double t = s_Data[pos] + s_Data[pos - offset];
    __syncthreads();
    s_Data[pos] = t;
  }
  return s_Data[pos];
}

inline __device__ double scan1Exclusive(double idata, volatile double *s_Data, uint size)
{
  return scan1Inclusive(idata, s_Data, size) - idata;
}

inline __device__ double4 scan4Inclusive(double4 idata4, volatile double *s_Data, uint size)
{
  //Level-0 inclusive scan
  idata4.y += idata4.x;
  idata4.z += idata4.y;
  idata4.w += idata4.z;

  //Level-1 exclusive scan
  double oval = scan1Exclusive(idata4.w, s_Data, size / 4);

  idata4.x += oval;
  idata4.y += oval;
  idata4.z += oval;
  idata4.w += oval;

  return idata4;
}

// Exclusive vector scan: the array to be scanned is stored
// in local thread memory scope as uint4
inline __device__ double4 scan4Exclusive(double4 idata4, volatile double *s_Data, uint size)
{
  double4 odata4 = scan4Inclusive(idata4, s_Data, size);
  odata4.x -= idata4.x;
  odata4.y -= idata4.y;
  odata4.z -= idata4.z;
  odata4.w -= idata4.w;

  return odata4;
}

__global__ void scanExclusiveShared(double4 *d_Dst, double4 *d_Src, uint size)
{
  __shared__ double s_Data[2 * THREADBLOCK_SIZE];
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data
  double4 idata4 = d_Src[pos];

  // Calculate exclusive scan
  double4 odata4 = scan4Exclusive(idata4, s_Data, size);

  // Write back
  d_Dst[pos] = odata4;
}

__global__ void scanInclusiveShared(double4 *d_Dst, double4 *d_Src, double *d_sum, uint size)
{
  __shared__ double s_Data[2 * THREADBLOCK_SIZE];
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data
  double4 idata4 = d_Src[pos];

  // Calculate exclusive scan
  double4 odata4 = scan4Inclusive(idata4, s_Data, size);

  // Write back
  d_Dst[pos] = odata4;

  if (pos == (size/4-1))
  {
    d_sum[0] = odata4.w;
  }

}

__global__ void scanInclusiveShared(double4 *d_Dst, double4 *d_Src, uint size)
{
  __shared__ double s_Data[2 * THREADBLOCK_SIZE];
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data
  double4 idata4 = d_Src[pos];

  // Calculate exclusive scan
  double4 odata4 = scan4Inclusive(idata4, s_Data, size);

  // Write back
  d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(double *d_Buf, double *d_Dst, double *d_Src, uint N, uint arrayLength)
{
  __shared__ double s_Data[2 * THREADBLOCK_SIZE];

  // Skip loads and stores for inactive threads of last threadblock (pos >= N)
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  // Load top elements
  // Convert results of bottom-level scan back to inclusive
  double idata = 0;

  if (pos < N)
    idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
      d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

  // Compute
  double odata = scan1Exclusive(idata, s_Data, arrayLength);

  // Avoid out-of-bound access
  if (pos < N)
  {
    d_Buf[pos] = odata;
  }
}

//inclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanInclusiveShared2(double *d_Buf, double *d_Dst, double *d_Src, uint N, uint arrayLength)
{
  __shared__ double s_Data[2 * THREADBLOCK_SIZE];

  //Skip loads and stores for inactive threads of last threadblock (pos >= N)
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  //Load top elements
  //Convert results of bottom-level scan back to inclusive
  double idata = 0;

  if (pos < N)
    idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
      d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

  //Compute
  double odata = scan1Inclusive(idata, s_Data, arrayLength);

  //Avoid out-of-bound access
  if (pos < N)
  {
    d_Buf[pos] = odata;
  }
}

__global__ void uniformUpdate(double4 *d_Data, double *d_Buffer, double* total, uint arrayLength)
{
  __shared__ double buf;
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0)
  {
    buf = d_Buffer[blockIdx.x];
  }

  __syncthreads();

  double4 data4 = d_Data[pos];
  data4.x += buf;
  data4.y += buf;
  data4.z += buf;
  data4.w += buf;
  d_Data[pos] = data4;

  if (pos == (arrayLength/4-1))
  {
    total[0] = data4.w;
  }
}

__global__ void normalizeWeightCDF(double *d_Dst, double* normalizer)
{
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;
  d_Dst[pos] /= normalizer[0];
}

__global__ void sampleParallel(double *d_weights_cdf, unsigned int *indices, double step,
                               double seed, uint len_i)
{
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;
  // get current step in iteration
  double cur_it = static_cast<double> (pos) * step + seed;
  uint i = static_cast<uint> (cur_it * static_cast<double>(len_i));

  if (pos < len_i)
  {
    // once the iterator is less than a weight, we have found the index
    double lb(0.0);
    if (i > 0)
      lb = d_weights_cdf[i - 1];
    else lb = 0.0;

    while (cur_it > d_weights_cdf[i] || cur_it < lb) {
      if (cur_it > d_weights_cdf[i])
        i++;
      else
        i--;

      if (i > 0)
        lb = d_weights_cdf[i - 1];
      else lb = 0.0;
    }
    indices[pos] = i;
  }
}


}  // namespace gpu_pf