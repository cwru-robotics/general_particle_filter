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

#include <gtest/gtest.h>
#include <vector>
#include <general_particle_filter/cpu/particle_filter.h>

#include <iostream>
#include <random>


struct resamplingTest : testing::Test, cpu_pf::ParticleFilter
{
    
    double manual_weights_[10];
    unsigned int sample_indecis_[10];
    double seed_;

    // test object constructor
    resamplingTest():
    ParticleFilter(10, 10),
    manual_weights_({0.05, 0.1, 0.05, 0.2, 0.15, 0.05, 0.1, 0.1, 0.05, 0.15}),
    sample_indecis_({0, 1, 3, 3, 4, 4, 6, 7, 8, 9}),
    seed_(0.03)
    {
    };

    ~resamplingTest()
    {
    }
};


TEST_F(resamplingTest, testResampling)
{
    // Import image
    // transform the pointList (and get the JAC)
    for (int i = 0; i < 10; i++)
    {
        setParticleWeight(i, manual_weights_[i]);
    }

    construct_weight_cdf();

    sampleParticles(seed_);

    for (unsigned int i = 0; i < 10; i++)
    {
        unsigned int i_d(getSampleIndex(i));
        printf("sampled index <%d, %d> \n", i_d, sample_indecis_[i]);
        ASSERT_EQ(i_d, sample_indecis_[i]);
    }
	}



int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}