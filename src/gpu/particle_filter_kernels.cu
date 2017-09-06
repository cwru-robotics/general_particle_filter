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
#define THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing
extern "C" const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;


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

inline __device__ uint scan1Exclusive(double idata, volatile double *s_Data, uint size)
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
  uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

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


__global__ void sampleParallel(double *weights, unsigned int *indices, double step,
                               double seed, int len)
{
  int idx = threadIdx.x;
  // get current step in iteration
  double cur_it = idx * step + seed;

  int i = 0;
  // once the iterator is less than a weight, we have found the index
  while (cur_it > weights[i])
  {
    i++;
  }
  indices[idx] = i;
}


}  // namespace gpu_pf