#include <general_particle_filter/gpu/particle_filter.h>

#include <gtest/gtest.h>
#include <vector>

#include <iostream>
#include <random>


struct resamplingTest : testing::Test, gpu_pf::ParticleFilterGPU
{

    double manual_weights_[10];
    unsigned int sample_indices_[10];
    double seed_;

    // test object constructor
    resamplingTest():
            ParticleFilterGPU(10, 10),
            manual_weights_({0.05, 0.1, 0.05, 0.2, 0.15, 0.05, 0.1, 0.1, 0.05, 0.15}),
            sample_indices_({0, 1, 3, 3, 4, 4, 6, 7, 8, 9}),
            seed_(0.03)
    {
    };

    ~resamplingTest()
    {
    }
};


TEST_F(resamplingTest, testResampling)
{

    for (int i = 0; i < 10; i++)
    {
        setParticleWeight(i, manual_weights_[i]);
    }

    construct_weight_cdf();

    sampleParticles(seed_);

    for (unsigned int i = 0; i < 10; i++)
    {
        unsigned int i_d(getSampleIndex(i));
        printf("sampled index <%d, %d> \n", i_d, sample_indices_[i]);
        ASSERT_EQ(i_d, sample_indices_[i]);
    }
}



int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
