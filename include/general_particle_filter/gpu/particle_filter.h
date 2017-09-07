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

#ifndef GENERAL_PARTICLE_FILTER_GPU_PARTICLE_FILTER_H
#define GENERAL_PARTICLE_FILTER_GPU_PARTICLE_FILTER_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <boost/random/normal_distribution.hpp>


#define THREADBLOCK_SIZE 512

namespace gpu_pf
{

extern const uint MAX_BATCH_ELEMENTS;
extern const uint MIN_SHORT_ARRAY_SIZE;
extern const uint MAX_SHORT_ARRAY_SIZE;
extern const uint MIN_LARGE_ARRAY_SIZE;
extern const uint MAX_LARGE_ARRAY_SIZE;


uint iDivUp(uint dividend, uint divisor);


/**
 * @brief GPU global function to compute the sample indices.
 *
 * This function is called by the cpu side code but is executed on the gpu side.
 *
 * @param weights vector of cdf weights
 * @param indices output vector of indices drawn in resampling
 * @param step defined as 1/# output particles
 * @param seed random seed number for the low variance resampling.
 * @param len # particles
 */
__global__ void sampleParallel(double * weights, unsigned int * indices, double step,
                               double seed, uint len);

/**
 * @brief The small variable sized
 * @param d_Dst
 * @param d_Src
 * @param total
 * @param size
 */
__global__ void scanInclusiveShared(double4 *d_Dst, double4 *d_Src, double* total, uint size);

__global__ void scanInclusiveShared(double4 *d_Dst, double4 *d_Src, uint size);

__global__ void scanExclusiveShared(double4 *d_Dst, double4 *d_Src, uint size);

__global__ void normalizeWeightCDF(double *d_Dst, double* normalizer);

__global__ void scanInclusiveShared2(double *d_Buf, double *d_Dst, double *d_Src, uint N, uint arrayLength);

__global__ void scanExclusiveShared2(double *d_Buf, double *d_Dst, double *d_Src, uint N, uint arrayLength);

__global__ void uniformUpdate(double4 *d_Data, double *d_Buffer, double *total, uint size);

/**
 * @brief An abstract class which implements the common features of a particle filter on GPU
 */
class ParticleFilter
{
public:
  /**
   * @brief The GPU particle filter constructor (explicitly defined)
   * @param n The number of weights to use for resampling.
   * @param m The number of entries to resample.
   */
  explicit ParticleFilter(int n, int m);

/**
 * @brief sets the weight of a particle
 * @param index of the weight to set.
 * @param value new value of the weight.
 */
  void setParticleWeight(int index, double value);

  /**
  * @brief constructs the cdf of the weights from the pdf on host side
  * The CDF ends at 1.0
  */
  void construct_weight_cdf();


  /**
  * @brief sample particles using the weights on device side in parallel and copy to host side
  * @param seed optional random iterator for sampling
  * It is expected that weight CDF exists and that all of the memory is allocated.
  * The sampling is completed using the low variance algorithm.
  */
  void sampleParticleIndecis(double seed = -1.0);

  /**
  * @brief gets the one index from the resample vector
  * @param index the Index of the sample
  * @return the sampled from index.
  */
  unsigned int getSampleIndex(unsigned int index);

  /**
   * @brief get the sampling length of the filter
   */
  int getSamplingSize() const;

  /**
   * @brief get the sampling length of the filter
   */
  int getPaddedSamplingSize() const;

  /**
   * @brief get the weight length of the filter
   */
  int getWeightsSize() const;

  /**
   * @brief get the sampling length of the filter
   */
  int getPaddedWeightsSize() const;

  int get4PaddedWeightsSize() const;

  /**
  * @brief the device side cumulative density function of the weights
  */
  double * d_weights_cdf_;

  /**
   * @brief the device side probability density function of the weights
   */
  double * d_weights_pdf_;

  /**
   * @brief the device side sampled indices computed given a set of weights
   */
  unsigned int * d_sample_indices_;

private:
  /**
   * @brief the probability density function of the weights host side.
   */
  double * h_weights_pdf_;

  /**
   * @brief the host side sampled indices computed given a set of weights
   */
  unsigned int * h_sample_indices_;

  /**
  * @brief the host cumulative density function of the weights.
  */
  double * h_weights_cdf_;

  /**
  * @brief the device sum;
  */
  double * d_weights_total_;

  /**
   * @brief The padded number of allocated weights.
   *
   * The padding is necessary for maintaining an optimized array size.
   * The extraneous weights are padded at the end weight array.
   */
  int allocated_samples_padded_;

  /**
   * @brief allocated status of the weights
   */
  int allocated_weights_;

  /**
  * @brief allocated size of the resampling
  */
  int allocated_samples_;

  /**
   * @brief The padded number of allocated weights.
   *
   * The padding is necessary for maintaining an optimized array size.
   * The extraneous sampled are padded at the end weight array.
   */
  int allocated_weights_padded_block_;


  int allocated_weights_padded_opt_;


  /**
   * @brief Allocates the vector of weights for use in the particle filter
   * Allocate pdf host side and cdf device/host side
   * @param n The length of the weight vector.
   * @return 0 on sucess error otherwise.
   */
  int allocateWeights(uint n);

  /**
   * @brief Allocates the vector of sample indices host/device side for use in the particle filter
   * @param n The length of the sample vector.
   * @return 0 on sucess error otherwise.
   */
  int allocateSamples(uint n);

  /**
   * @brief normalize the weights of the particle filter such that the weights sum to 1.
   * This assumes no CDF and that the weights are already allocated.
   * @param denominator optional precomputed denominator.
   */
  void normalize_weights(double denominator = -1.0);

  /**
   * @brief deallocates the device/host vectors of samples for use in the particle filter
   */
  void deallocateSamples();

  /**
   * @brief deallocates the device/host vectors of weights for use in the particle filter
   */
  void deallocateWeights();
};  // class gpu_pf::ParticleFilter


}  // namespace gpu_pf



#endif  // GENERAL_PARTICLE_FILTER_GPU_PARTICLE_FILTER_H
