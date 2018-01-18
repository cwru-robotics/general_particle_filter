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

#include <general_particle_filter/gpu/particle_filter.h>

#include <gtest/gtest.h>
#include <vector>

#include <iostream>
#include <random>


struct resamplingTest : testing::Test, gpu_pf::ParticleFilter
{
  // test object constructor
  resamplingTest(): ParticleFilter(10, 10)
  {
  };

  ~resamplingTest()
  {
  };
};


TEST_F(resamplingTest, testResampling)
{
  double manual_weights[10] = {0.05, 0.1, 0.05, 0.2, 0.15, 0.05, 0.1, 0.1, 0.05, 0.15};
  unsigned int sample_indices[10] = {0, 1, 3, 3, 4, 4, 6, 7, 8, 9};
  double seed(0.03);
  for (int i = 0; i < 10; i++)
  {
    setParticleWeight(i, manual_weights[i]);
  }

  construct_weight_cdf();

  sampleParticlesIndecis(seed);

  for (unsigned int i = 0; i < 10; i++)
  {
    unsigned int i_d(getSampleIndex(i));
    printf("sampled index <%d, %d> \n", i_d, sample_indices[i]);
    ASSERT_EQ(i_d, sample_indices[i]);
  }
}

TEST_F(resamplingTest, testCDF)
{
  double manual_weights[10] = {0.05, 0.1, 0.05, 0.2, 0.15, 0.05, 0.1, 0.1, 0.05, 0.15};
  unsigned int sample_indices[10] = {0, 1, 3, 3, 4, 4, 6, 7, 8, 9};
  double seed(0.03);
  for (int i = 0; i < 10; i++)
  {
    setParticleWeight(i, manual_weights[i]);
  }

  construct_weight_cdf();

  sampleParticles(seed);

  for (unsigned int i = 0; i < 10; i++)
  {
    unsigned int i_d(getSampleIndex(i));
    printf("sampled index <%d, %d> \n", i_d, sample_indices[i]);
    ASSERT_EQ(i_d, sample_indices[i]);
  }
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
