#include <general_particle_filter/gpu/particle_filter.h>
#include <iostream>
#include <cuda.h>
#include <ros/ros.h>


int main(int argc, char **argv)
{
    gpu_pf::ParticleFilterGPU pf(10, 10);

    double weights[10] = {1,2,3,4,5,6,7,8,9,10};
    double h_weights[10] = {0.1, 0.2, 0.05, 0.1, 0.05, 0.05, 0.15, 0.2, .05, 0.05};

    for (int i=0; i<10;i++)
        pf.setParticleWeight(i, h_weights[i]);
    pf.construct_weight_cdf();

    pf.sampleParticles(0.05);

    return 0;
}