#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <cuda_runtime.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <boost/random/normal_distribution.hpp>


namespace gpu_pf {

/**
 * @brief An abstract class which implements the common features of a particle filter on GPU
 */
class ParticleFilterGPU {

public:

    /**
     * @brief The GPU particle filter constructor (explicitly defined)
     * @param n The number of weights to use for resampling.
     */
    explicit ParticleFilterGPU(int n, int m);

  /**
   * @brief sets the weight of a particle
   * @param index of the weight to set.
   * @param new value of the weight.
   */
    void setParticleWeight(int index, double value);

    /**
    * @brief constructs the cdf of the weights from the pdf on host side
    * The CDF ends at 1.0
    * @param denominator" optional precomputed denominator.
    */
    void construct_weight_cdf(double denominator = -1.0);

    /**
    * @brief sample particles using the weights on device side in parallel and copy to host side
    * @param seed optional random iterator for sampling
    * It is expected that weight CDF exists and that all of the memory is allocated.
    * The sampling is completed using the low variance algorithm.
    */
    void sampleParticles(double seed = -1.0);

    /**
    * @brief gets the one index from the resample vector
    * @param the Index of the sample
    * @return the sampled from index.
    */
    unsigned int getSampleIndex(unsigned int index);

    /**
    * @brief the device side cumulative density function of the weights
    */
    double * d_weights_cdf_;

    /**
     * @brief the device side sampled indices computed given a set of weights
     */
    unsigned int * d_sample_indices_;


private:

    /**
     * @brief Allocates the vector of weights for use in the particle filter
     * Allocate pdf host side and cdf device/host side
     * @param The length of the weight vector.
     * @return 0 on sucess error otherwise.
     */
    int allocateWeights(int n);

    /**
     * @brief Allocates the vector of sample indices host/device side for use in the particle filter
     * @param The length of the sample vector.
     * @return 0 on sucess error otherwise.
     */
    int allocateSamples(int n);

    /**
     * @brief normalize the weights of the particle filter such that the weights sum to 1.
     * This assumes no CDF and that the weights are already allocated.
     * @param optional precomputed denominator.
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

    /**
     * @brief number of bytes used given the number of particles for arrays of doubles
     * Used for memory allocation
     */
    int double_bytes;

    /**
     * @brief number of bytes used given the number of particles for arrays of ints
     * Used for memory allocation
     */
    int int_bytes;

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
     * @brief allocated status of the weights
     */
    int allocated_weights_;

    /**
    * @brief allocated size of the resampling
    */
    int allocated_samples_;


}; //class <ParticleFilterGPU

} //namespace gpu_pf

/**
 * @brief GPU helper function to compute one sample index
 * @param weights vector of cdf weights
 * @param indices output vector of indices drawn in resampling
 * @param step defined as 1/# particles
 * @param seed random iterator
 * @param len # particles
 * @return
 */
__global__ void sampleParallel(double * weights, unsigned int * indices, double step,
                              double seed, int len );



#endif //GENERAL_PARTICLE_FILTER_GENERAL_PARTICLE_FILTER_H
