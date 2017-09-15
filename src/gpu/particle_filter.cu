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
#include "general_particle_filter/gpu/particle_filter.h"

namespace gpu_pf
{

uint factorRadix2(uint &log2L, uint L)
{
  if (!L)
  {
    log2L = 0;
    return 0;
  }
  else
  {
    for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);

    return L;
  }
}


inline uint pow2roundup (uint x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x+1;
}

uint iDivUp(uint dividend, uint divisor)
{
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

ParticleFilter::ParticleFilter(int n, int m) :
  d_weights_cdf_(NULL),
  d_weights_pdf_(NULL),
  h_weights_cdf_(NULL),
  h_weights_pdf_(NULL),
  d_weights_total_(NULL),
  allocated_weights_(0),
  allocated_weights_padded_block_(0),
  allocated_weights_padded_opt_(0),
  allocated_samples_(0)
{
  allocateWeights(n);
  allocateSamples(m);
}

int ParticleFilter::allocateWeights(uint n)
{

  uint multiplier4(iDivUp(n, THREADBLOCK_SIZE * 4));
  uint multiplier(iDivUp(n, THREADBLOCK_SIZE));


  uint n_padded_opt_a(4 * THREADBLOCK_SIZE * multiplier4);
  uint n_padded_block(THREADBLOCK_SIZE * multiplier);


  uint log2L;
  uint factorizationRemainder(factorRadix2(log2L, n_padded_opt_a));

  uint n_padded_opt(pow2roundup(n_padded_opt_a));

  /*if (factorizationRemainder > 1)
  {
    n_padded_opt = 2 << (log2L + 1);
  }*/

  size_t allocation_size(static_cast<size_t>(n_padded_opt));

  h_weights_pdf_ = reinterpret_cast<double *> (calloc(allocation_size, sizeof(double)));
  h_weights_cdf_ = reinterpret_cast<double *> (calloc(allocation_size, sizeof(double)));

  cudaError_t  success_01(cudaMalloc((void **) &d_weights_cdf_, allocation_size * sizeof(double)));
  cudaError_t  success_02(cudaMalloc((void **) &d_weights_pdf_, allocation_size * sizeof(double)));

  cudaMemcpy(d_weights_pdf_, h_weights_pdf_,  sizeof(double) * allocation_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights_cdf_, h_weights_cdf_,  sizeof(double) * allocation_size, cudaMemcpyHostToDevice);

  cudaError_t  success_03(cudaMalloc((void **) &d_weights_total_, sizeof(double)));

  allocated_weights_ = n;
  allocated_weights_padded_block_ = n_padded_block;
  allocated_weights_padded_opt_ = n_padded_opt;

  printf("Allocated weight size: <%d, %d, %d, %d>\n", allocated_weights_, allocated_weights_padded_block_,
    n_padded_opt_a, allocated_weights_padded_opt_);

  if (success_01 == cudaSuccess && success_02 == cudaSuccess)
    return 0;
  else
    return -1;
}


int ParticleFilter::allocateSamples(uint n)
{
  size_t allocation_size(static_cast<size_t>(n));

  uint n_padded(iDivUp(n , THREADBLOCK_SIZE) * THREADBLOCK_SIZE);


  h_sample_indices_ = reinterpret_cast<unsigned int *> (calloc(n_padded, sizeof(unsigned int)));

  cudaError_t  success(cudaMalloc((void **) &d_sample_indices_, n_padded * sizeof(unsigned int)));

  cudaMemcpy(d_sample_indices_, h_sample_indices_,  sizeof(unsigned int) * n_padded, cudaMemcpyHostToDevice);

  printf("allocated sample size: <%d, %d>\n", n, n_padded);
  allocated_samples_ = n;
  allocated_samples_padded_ = n_padded;
  if (success == cudaSuccess)
    return 0;
  else
    return -1;
}

void ParticleFilter::setParticleWeight(int index, double weights)
{
  h_weights_pdf_[index] = weights;
}


void ParticleFilter::construct_weight_cdf()
{
  uint arrayLength(allocated_weights_padded_opt_);
  uint batchSize(1);

  //Check power-of-two factorization
  uint log2L;
  uint factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  //Check total batch size limit
  assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

  if (arrayLength > MAX_SHORT_ARRAY_SIZE)
  {
    double * d_Buf;

    cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(double));

    // long array analysis
    //Check supported size range
    assert((arrayLength <= MAX_LARGE_ARRAY_SIZE));

    scanInclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
      reinterpret_cast<double4 *>(d_weights_cdf_),
      reinterpret_cast<double4 *>(d_weights_pdf_), 4 * THREADBLOCK_SIZE);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Scan Inclusive Error: %s\n", cudaGetErrorString(err));

    // Not all threadblocks need to be packed with input data:
    // inactive threads of highest threadblock just don't do global reads and writes
    const uint blockCount2 = iDivUp((batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);

    scanExclusiveShared2<<< blockCount2, THREADBLOCK_SIZE>>>(
        d_Buf, d_weights_cdf_, d_weights_pdf_,
        (batchSize *arrayLength) / (4 * THREADBLOCK_SIZE),
        arrayLength / (4 * THREADBLOCK_SIZE)
    );
    err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Shared Scan 2 Error: %s\n", cudaGetErrorString(err));

    uniformUpdate<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
      reinterpret_cast<double4*>(d_weights_cdf_), d_Buf, d_weights_total_, arrayLength);

    err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(err));

    cudaFree(d_Buf);
  }
  else
  {
    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE));

    // Check all threadblocks to be fully packed with data
    assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);

    scanInclusiveShared<<<1, THREADBLOCK_SIZE>>>(
      reinterpret_cast<double4*>(d_weights_cdf_), reinterpret_cast<double4*>(d_weights_pdf_),
      d_weights_total_, arrayLength);
  }
  normalizeWeightCDF<<<arrayLength/THREADBLOCK_SIZE, THREADBLOCK_SIZE>>>(d_weights_cdf_, d_weights_total_);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
}

void ParticleFilter::sampleParticleIndecis(double seed)
{
  double i_seed(seed); //initialize random iterator
  double sample_interval(1.0 / static_cast<double>(allocated_samples_)); //declare sample interval as 1/#particles
  if (i_seed < 0.0 || i_seed >= sample_interval) //generate random iterator
  {
    srand((unsigned) time(NULL));
    double random = ((double) rand()) / (double) RAND_MAX;  //random # between 0-1
    i_seed = (random * -sample_interval) + sample_interval; //random # between 0 and max
  }
  uint block_padded(iDivUp(allocated_samples_, THREADBLOCK_SIZE));
  sampleParallel <<< block_padded, THREADBLOCK_SIZE>>> (d_weights_cdf_, d_sample_indices_,
    sample_interval, i_seed, allocated_samples_);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Compute Sample Indecis Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(h_sample_indices_, d_sample_indices_, allocated_samples_ * sizeof(uint), cudaMemcpyDeviceToHost); //copy output to host
}

void ParticleFilter::deallocateSamples()
{
  free(h_sample_indices_);
  h_sample_indices_ = NULL;
  allocated_samples_ = 0;
  cudaFree(d_sample_indices_);
}

void ParticleFilter::deallocateWeights()
{
  free(h_weights_cdf_);
  h_weights_cdf_ = NULL;
  free(h_weights_pdf_);
  h_weights_pdf_ = NULL;
  allocated_weights_ = 0;
  cudaFree(d_weights_cdf_);
  cudaFree(d_weights_pdf_);
  cudaFree(d_weights_total_);
}

unsigned int ParticleFilter::getSampleIndex(unsigned int index)
{
  return h_sample_indices_[index];
}

int ParticleFilter::getSamplingSize() const
{
  return allocated_samples_;
}

int ParticleFilter::getWeightsSize() const
{

  return allocated_weights_;
}

int ParticleFilter::getPaddedSamplingSize() const
{
  return allocated_samples_padded_;
}

int ParticleFilter::getPaddedWeightsSize() const
{
  return allocated_weights_padded_block_;
}

int ParticleFilter::get4PaddedWeightsSize() const
{
  return allocated_weights_padded_opt_;
}




}  // namespace gpu_pf



