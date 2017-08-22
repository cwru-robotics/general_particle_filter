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

ParticleFilter::ParticleFilter(int n, int m) :
  d_weights_cdf_(NULL),
  d_weights_pdf_(NULL),
  h_weights_cdf_(NULL),
  h_weights_pdf_(NULL),
  double_bytes(n * sizeof(double)),
  int_bytes(n * sizeof(unsigned int)),
  allocated_weights_(0),
  allocated_samples_(0)
{
  allocateWeights(n);
  allocateSamples(m);
}

int ParticleFilter::allocateWeights(int n)
{

  size_t allocation_size(static_cast<size_t>(n));
  h_weights_pdf_ = reinterpret_cast<double *> (calloc(allocation_size, sizeof(double)));
  h_weights_cdf_ = reinterpret_cast<double *> (calloc(allocation_size, sizeof(double)));

  cudaError_t  success_01(cudaMalloc((void **) &d_weights_cdf_, double_bytes));
  cudaError_t  success_02(cudaMalloc((void **) &d_weights_pdf_, double_bytes));

  allocated_weights_ = n;

  if (success_01 == cudaSuccess && success_02 == cudaSuccess)
    return 0;
  else
    return -1;
}

int ParticleFilter::allocateSamples(int n)
{
  size_t allocation_size(static_cast<size_t>(n));
  h_sample_indices_ = reinterpret_cast<unsigned int *> (calloc(allocation_size, sizeof(unsigned int)));

  cudaError_t  success(cudaMalloc((void **) &d_sample_indices_, int_bytes));

  allocated_samples_ = n;
  if (success == cudaSuccess)
    return 0;
  else
    return -1;
}

void ParticleFilter::setParticleWeight(int index, double weights)
{
  h_weights_pdf_[index] = weights;
}


void ParticleFilter::construct_weight_cdf(double denominator)
{
  if (allocated_weights_ == 0) return;

  // first normalize the weights
  normalize_weights(denominator);

  // Set first weight, note that this is an inclusive cumulative sum.

  h_weights_cdf_[0] = h_weights_pdf_[0];

  //Set remaining weights as sum of all elements before
  for (int i(1); i < allocated_weights_; i++)
  {
    h_weights_cdf_[i] = h_weights_cdf_[i - 1] + h_weights_pdf_[i];
  }


  cudaMemcpy(d_weights_cdf_, h_weights_cdf_, double_bytes, cudaMemcpyHostToDevice);
}

void ParticleFilter::normalize_weights(double denominator)
{
  if (allocated_weights_ == 0) return;

  double denom(denominator);
  if (denom <= 0) //sum all weights
  {
    denom = 0.0;
    for (int i(0); i < allocated_weights_; i++)
    {
      denom += h_weights_pdf_[i];
    }
  }

  for (int i(0); i < allocated_weights_; i++)
  {
    h_weights_pdf_[i] /= denom; // divide each element by denominator
  }
}

void ParticleFilter::sampleParticles(double seed)
{
  double i_seed(seed); //initialize random iterator
  double sample_interval(1.0 / static_cast<double>(allocated_samples_)); //declare sample interval as 1/#particles
  if (i_seed < 0.0 || i_seed >= sample_interval) //generate random iterator
  {
    srand((unsigned) time(NULL));
    double random = ((double) rand()) / (double) RAND_MAX;  //random # between 0-1
    i_seed = (random * -sample_interval) + sample_interval; //random # between 0 and max
  }

  sampleParallel << < 1, allocated_samples_ >> > (d_weights_cdf_, d_sample_indices_,
    sample_interval, i_seed, allocated_weights_); //get all sampled indices in parallel

  cudaMemcpy(h_sample_indices_, d_sample_indices_, int_bytes, cudaMemcpyDeviceToHost); //copy output to host
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





}  // namespace gpu_pf



