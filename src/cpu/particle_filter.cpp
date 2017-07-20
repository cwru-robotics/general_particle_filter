/*
 *  particle_filter_cpu.cpp
 *  Copyright (C) 2017  Russell Jackson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <general_particle_filter/cpu/particle_filter.h>
#include <stdlib.h> 
#include <random>

namespace cpu_pf
{


ParticleFilterCPU::ParticleFilterCPU(int n, int m):
weights_cdf_(NULL),
weights_pdf_(NULL),
allocated_weights_(0),
allocated_samples_(0)
{
  allocateWeights(n);
  allocateSamples(m);
}


ParticleFilterCPU::~ParticleFilterCPU()
{
  deallocateWeights();
  deallocateSamples();
}

void ParticleFilterCPU::normalize_weights(double denominator)
{
  if (allocated_weights_ == 0) return;

  double denom(denominator);
  if (denom <= 0)
  {
    denom = 0.0;
    for (int i(0); i < allocated_weights_; i++)
    {
      denom += weights_pdf_[i];
    }
  }

  for (int i(0); i < allocated_weights_; i++)
  {
    weights_pdf_[i] /= denom;
  }
}

void ParticleFilterCPU::construct_weight_cdf(double denominator)
{
  if (allocated_weights_ == 0) return;
  // first normalize the weights
  normalize_weights(denominator);

  weights_cdf_[0] = weights_pdf_[0];

  for (int i(1); i < allocated_weights_; i++)
  {
    weights_cdf_[i] = weights_cdf_[i-1] + weights_pdf_[i];
  }
}

int ParticleFilterCPU::allocateWeights(int n)
{
  size_t allocation_size(static_cast<size_t>(n));
  weights_cdf_ = reinterpret_cast<double*> (calloc(allocation_size, sizeof(double)));
  weights_pdf_ = reinterpret_cast<double*> (calloc(allocation_size, sizeof(double)));
  allocated_weights_ = n;
}

int ParticleFilterCPU::allocateSamples(int m)
{
  size_t allocation_size(static_cast<size_t>(m));
  sample_indecis_ = reinterpret_cast<unsigned int*> (calloc(allocation_size, sizeof(unsigned int)));
  
  allocated_samples_ = m;
}

void ParticleFilterCPU::deallocateWeights()
{
  free (weights_cdf_);
  weights_cdf_ = NULL;
  free (weights_pdf_);
  weights_pdf_ = NULL;
  allocated_weights_ = 0;
}

void ParticleFilterCPU::deallocateSamples()
{
  free (sample_indecis_);
  sample_indecis_ = NULL;
  allocated_samples_ = 0;
}

void ParticleFilterCPU::sampleParticles(double seed)
{
  double i_seed(seed);
  double sample_interval(1.0/static_cast<double>(allocated_samples_));
  if (i_seed < 0.0 || i_seed >= sample_interval)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0,sample_interval);
    i_seed = distribution(gen);
  }

  // sample higher weighted particles into result_points
  unsigned int weight_index(0);
  for (unsigned int index(0); index < allocated_samples_; index++)
  {
    double index_f = i_seed + static_cast<double> (index) * sample_interval;

    while (index_f > weights_cdf_[weight_index])
    {
      weight_index += 1;
    }
    sample_indecis_[index] = weight_index;
  }
}

void ParticleFilterCPU::setParticleWeight(int index, double value)
{
  weights_pdf_[index] = value;
}

unsigned int ParticleFilterCPU::getSampleIndex(unsigned int index)
{
  return sample_indecis_[index];
}


};  // namespace cpu_pf