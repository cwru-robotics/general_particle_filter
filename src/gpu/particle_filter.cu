#include <general_particle_filter/gpu/particle_filter.h>


namespace gpu_pf {

ParticleFilterGPU::ParticleFilterGPU(int n, int m):
    d_weights_cdf_(NULL),
    h_weights_cdf_(NULL),
    h_weights_pdf_(NULL),
    double_bytes(n*sizeof(double)),
    int_bytes(n*sizeof(unsigned int)),
    allocated_weights_(0),
    allocated_samples_(0)
{
    allocateWeights(n);
    allocateSamples(m);
}


int ParticleFilterGPU::allocateWeights(int n)
{

    size_t allocation_size(static_cast<size_t>(n));
    h_weights_pdf_ = reinterpret_cast<double*> (calloc(allocation_size, sizeof(double)));
    h_weights_cdf_ = reinterpret_cast<double*> (calloc(allocation_size, sizeof(double)));

    cudaMalloc((void **) &d_weights_cdf_, double_bytes);

    allocated_weights_ = n;
}

int ParticleFilterGPU::allocateSamples(int n)
{
    size_t allocation_size(static_cast<size_t>(n));
    h_sample_indices_ = reinterpret_cast<unsigned int*> (calloc(allocation_size, sizeof(unsigned int)));

    cudaMalloc((void **) &d_sample_indices_, int_bytes);

    allocated_samples_ = n;
}

void ParticleFilterGPU::setParticleWeight(int index, double weights)
{
    h_weights_pdf_[index] = weights;
}

void ParticleFilterGPU::construct_weight_cdf(double denominator)
{
    if (allocated_weights_ == 0) return;

    // first normalize the weights
    normalize_weights(denominator);

    //Set first weight
    h_weights_cdf_[0] = h_weights_pdf_[0];

    //Set remaining weights as sum of all elements before
    for (int i(1); i < allocated_weights_; i++)
    {
        h_weights_cdf_[i] = h_weights_cdf_[i-1] + h_weights_pdf_[i];
    }

    cudaMemcpy(d_weights_cdf_, h_weights_cdf_, double_bytes, cudaMemcpyHostToDevice);
}

void ParticleFilterGPU::normalize_weights(double denominator)
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

void ParticleFilterGPU::sampleParticles(double seed)
{
    double i_seed(seed); //initialize random iterator
    double sample_interval(1.0 / static_cast<double>(allocated_samples_)); //declare sample interval as 1/#particles
    if (i_seed < 0.0 || i_seed >= sample_interval) //generate random iterator
    {
        srand((unsigned) time(NULL));
        double random = ((double) rand()) / (double) RAND_MAX;  //random # between 0-1
        i_seed = (random * -sample_interval) + sample_interval; //random # between 0 and max
    }

    sampleParallel <<< 1, allocated_samples_ >>>(d_weights_cdf_, d_sample_indices_,
            sample_interval, i_seed, allocated_weights_); //get all sampled indices in parallel

    cudaMemcpy(h_sample_indices_, d_sample_indices_, int_bytes , cudaMemcpyDeviceToHost); //copy output to host

}
    
void ParticleFilterGPU::deallocateSamples()
{
    free (h_sample_indices_);
    h_sample_indices_ = NULL;
    allocated_samples_ = 0;
    cudaFree(d_sample_indices_);
}

void ParticleFilterGPU::deallocateWeights() {
    free(h_weights_cdf_);
    h_weights_cdf_ = NULL;
    free (h_weights_pdf_);
    h_weights_pdf_ = NULL;
    allocated_weights_ = 0;
    cudaFree(d_weights_cdf_);
}

unsigned int ParticleFilterGPU::getSampleIndex(unsigned int index)
{
        return h_sample_indices_[index];
    }


}

__global__ void sampleParallel(double * weights, unsigned int * indices, double step,
                               double seed, int len ) {
    int idx = threadIdx.x;
    double cur_it = idx * step + seed; //get current step in iteration

    int i = 0;
    while (cur_it > weights[i]) // once the iterator is less than a weight, we have found the index
    {
        i++;
    }
    indices[idx] = i;
}
