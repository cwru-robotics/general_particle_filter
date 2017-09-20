//
// Created by connor on 10/20/17.
//

#include "test_wrapper.h"

#include <general_particle_filter/gpu/particle_filter.h>



std::vector<int> test_resample(std::vector<double> weights, double seed)
{
    gpu_pf::ParticleFilter dummy_filter(weights.size(), weights.size());

    return std::vector<int>();
}