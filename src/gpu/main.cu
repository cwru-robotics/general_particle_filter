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
#include <iostream>
#include <cuda.h>
#include <ros/ros.h>


int main(int argc, char **argv)
{
    gpu_pf::ParticleFilter pf(10, 10);

    // double weights[10] = {1,2,3,4,5,6,7,8,9,10};
    double h_weights[10] = {0.1, 0.2, 0.05, 0.1, 0.05, 0.05, 0.15, 0.2, .05, 0.05};

    for (int i=0; i<10;i++)
        pf.setParticleWeight(i, h_weights[i]);
    pf.construct_weight_cdf();

    pf.sampleParticleIndecis(0.05);

    return 0;
}