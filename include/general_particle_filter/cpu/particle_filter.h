/*
 *  particle_filter.h
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

/*
 * The particle_filter.h defines an abstract base that can be used as a template for making fully functional particle filters
 */


#ifndef GENERAL_PARTICLE_FILTER_CPU_PARTICLE_FILTER_H
#define GENERAL_PARTICLE_FILTER_CPU_PARTICLE_FILTER_H

namespace cpu_pf
{

class ParticleFilter
{
public:

  /**
   * @brief The particle filter constructor (explicitly defined)
   *
   * @param The number of weights to use for resampling.
   * @param The number of resamples taken from the weight vector.
   */
  explicit ParticleFilter(int n, int m);

  
  ~ParticleFilter();
  /**
   * @brief constructs the cdf of the weights from the pdf.
   *
   * The CDF ends at 1.0
   *
   * @param optional precomputed denominator.
   */
  void construct_weight_cdf(double denominator = -1.0);

  /**
   * @brief sample particles using the weights cdf
   *
   * This function is entirely internal, i.e. no inputs and outputs.
   * It is expected that weight CDF exists and that all of the memory is allocated.
   * The sampling is completed using the low variance algorithm.
   */
  void sampleParticles(double seed = -1.0);

  /**
   * @brief sets the weight of a particle
   *
   * @param index of the weight to set. There is no check on the index (for perfomance)
   * @param new value of the weight.
   */
  void setParticleWeight(int index, double value);

  /**
   * @brief gets the sampled index of the resample vector
   *
   * @param the Index of the sample
   *
   * @return the sampled from index.
   */
  unsigned int getSampleIndex(unsigned int index);

  
  /**
   * @brief get the sampling length of the filter
   */
  int getSamplingSize() const;

  /**
   * @brief get the weight length of the filter
   */
  int getWeightsSize() const;

private:
  /**
   * @brief normalize the weights of the particle filter such that the weights sum to 1.
   *
   * This assumes no CDF and that the weights are already allocated.
   *
   * @param optional precomputed denominator.
   */
  void normalize_weights(double denominator = -1.0);

  /**
   * @brief Allocates the vector of weights for use in the particle filter
   *
   * @param The length of the weight vector.
   *
   * @return 0 on sucess error otherwise.
   */ 
  int allocateWeights(int n);

  /**
   * @brief Allocates the vector of sample indecis for use in the particle filter
   *
   * @param The length of the sample vector.
   *
   * @return 0 on sucess error otherwise.
   */ 
  int allocateSamples(int n);

  /**
   * @brief deallocates the vector of weights for use in the particle filter
   */
  void deallocateWeights();

  /**
   * @brief deallocates the vector of samples for use in the particle filter
   */
  void deallocateSamples();

  /**
   * @brief the cumulative density function of the weights.
   */
  double * weights_cdf_;

  /**
   * @brief the probability density function of the weights.
   */
  double * weights_pdf_;

  /**
   * @brief the cumulative density function of the weights.
   */
  unsigned int * sample_indecis_;

  /**
   * @brief allocated status of the weights 
   */
  int allocated_weights_;

  /**
   * @brief allocated size of the resampling
   */
  int allocated_samples_;

};

};  // namespace cpu_pf

#endif  // GENERAL_PARTICLE_FILTER_CPU_PARTICLE_FILTER_H
