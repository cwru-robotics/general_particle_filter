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

#include <random>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ros/ros.h>
#include "general_particle_filter/example/gpu_example_pf.h"
#include "general_particle_filter/example/planar_object_gpu.h"


namespace example_gpu_pf
{


PlanarParticleFilter::PlanarParticleFilter(int particle_count, const objectVariance &object_variance) :
    gpu_pf::ParticleFilter(particle_count, particle_count),
    object_variance_(object_variance)
{
  // allocate the internal state memory.
  allocateStates();
}

PlanarParticleFilter::~PlanarParticleFilter()
{
  deallocateStates();
}

void PlanarParticleFilter::allocateStates()
{
  int temp_count(this->getWeightsSize());
  int output_count(this->getSamplingSize());

  ObjectState* temp_particles = new ObjectState[temp_count];
  ObjectState* output_particles = new ObjectState[output_count];

  // allocate the gpu mem and perform the copy operator... (not sure what the best way is)
  cudaError_t  success_01(cudaMalloc(&d_temp_particles_, sizeof(ObjectState) * temp_count));
  cudaError_t  success_02(cudaMalloc(&d_output_particles_, sizeof(ObjectState) * output_count));

  cudaMalloc(&d_obj_action_, sizeof(objectAction));
  cudaMalloc(&d_obj_var_,   sizeof(objectVariance));

  objectAction temp_action(0.0, 0.0);

  cudaMemcpy(d_obj_action_, &temp_action, sizeof(objectAction), cudaMemcpyHostToDevice);
  cudaMemcpy(d_obj_var_, &object_variance_, sizeof(objectVariance), cudaMemcpyHostToDevice);

  cudaMemcpy(d_temp_particles_, temp_particles, sizeof(ObjectState) * temp_count, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output_particles_, output_particles, sizeof(ObjectState) * output_count, cudaMemcpyHostToDevice);

  delete [] temp_particles;
  delete [] output_particles;
}

void PlanarParticleFilter::deallocateStates()
{
  cudaFree(d_temp_particles_);
  cudaFree(d_output_particles_);
  cudaFree(d_obj_var_);
  cudaFree(d_obj_action_);
}

void PlanarParticleFilter::applyAction(const objectAction &object_action)
{
  int temp_count(this->getWeightsSize());
  int output_count(this->getSamplingSize());

  double * d_rand;
  cudaMalloc(&d_rand, temp_count * 2 * sizeof(double));
  curandGenerator_t prngGPU;
  curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
  std::random_device rd;

  curandSetPseudoRandomGeneratorSeed(prngGPU, rd());

  curandGenerateNormalDouble(prngGPU, d_rand, temp_count * 2, 0.0, 1.0);

  printf("applying the action <%3.3f, %3.3f> \n", object_action.dx_, object_action.dy_);
  cudaMemcpy(d_obj_action_, &object_action, sizeof(objectAction), cudaMemcpyHostToDevice);

  applyActionKernel<<<temp_count, 1>>>(d_obj_action_, d_rand, d_output_particles_, d_temp_particles_, d_obj_var_);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  ObjectState* local_object_mem_01 = reinterpret_cast<ObjectState*> (malloc(sizeof(ObjectState) * temp_count));
  ObjectState* local_object_mem_02 = reinterpret_cast<ObjectState*> (malloc(sizeof(ObjectState) * output_count));
  cudaMemcpy(local_object_mem_01, d_temp_particles_, sizeof(ObjectState) * temp_count, cudaMemcpyDeviceToHost);
  cudaMemcpy(local_object_mem_02, d_output_particles_, sizeof(ObjectState) * output_count, cudaMemcpyDeviceToHost);

  double * h_rand = reinterpret_cast<double*> (malloc(sizeof(double) * 2 * temp_count));
  cudaMemcpy(h_rand, d_rand, sizeof(double) * 2 * temp_count, cudaMemcpyDeviceToHost);

  objectAction gpu_action;
  cudaMemcpy(&gpu_action, d_obj_action_, sizeof(objectAction), cudaMemcpyDeviceToHost);
  printf("applied the action <%3.3f, %3.3f> \n", gpu_action.dx_, gpu_action.dy_);

  for (int index(0); index < 5; index++)
  {
    ROS_INFO("The output position is <%3.3f, %3.3f>", local_object_mem_02[index].x(), local_object_mem_02[index].y());
    ROS_INFO("The temp position is <%3.3f, %3.3f>", local_object_mem_01[index].x(), local_object_mem_01[index].y());
    ROS_INFO("The action rand is <%3.3f, %3.3f>", h_rand[index*2], h_rand[index*2 + 1]);

  }
  free(local_object_mem_01);
  free(local_object_mem_02);
  free(h_rand);
  cudaFree(d_rand);

}

void PlanarParticleFilter::applyObservation(const ObjectState &object_observation)
{
  int temp_count(this->getWeightsSize());

  computeParticleWeights<<<1, temp_count>>> (d_temp_particles_, d_weights_pdf_, object_observation, object_variance_);
  double * h_weights_cdf = reinterpret_cast<double*> (malloc(sizeof(double) * temp_count));
  cudaMemcpy(h_weights_cdf, d_weights_cdf_, sizeof(double) * temp_count, cudaMemcpyDeviceToHost);

  ROS_INFO("The observed position is <%3.3f, %3.3f>", object_observation.x(), object_observation.y());
  for (int i(0); i < temp_count; i++)
  {
    ROS_INFO("The weight at index %d is %f", i, h_weights_cdf[i]);
  }

  free(h_weights_cdf);

  this->construct_weight_cdf();
  this->sampleParticles();
}

ObjectState PlanarParticleFilter::estimateState()
{
  int output_count_i(this->getSamplingSize());
  double output_count_d(static_cast<double>(output_count_i));

  ObjectState* local_object_mem = reinterpret_cast<ObjectState*> (malloc(sizeof(ObjectState) * output_count_i));
  cudaMemcpy(local_object_mem, d_output_particles_, sizeof(ObjectState) * output_count_i, cudaMemcpyDeviceToHost);
  double x(0.0), y(0.0);
  for (int index(0); index < output_count_i; index++)
  {
    x += local_object_mem[index].x()/output_count_d;
    y += local_object_mem[index].y()/output_count_d;
    if (index == 0)
    {
      ROS_INFO("The aggregated position is <%3.3f, %3.3f>", local_object_mem[index].x(), local_object_mem[index].y());
    }
  }
  free(local_object_mem);
  return ObjectState(x, y);
}

visualization_msgs::MarkerArray PlanarParticleFilter::getParticleArray(const std_msgs::Header &header,
  const std_msgs::ColorRGBA &color)
{
  int output_count(this->getSamplingSize());
  ObjectState* local_object_mem = reinterpret_cast<ObjectState*> (malloc(sizeof(ObjectState) * output_count));
  cudaMemcpy(local_object_mem, d_output_particles_, sizeof(ObjectState) * output_count, cudaMemcpyDeviceToHost);

  visualization_msgs::MarkerArray outputArray;
  outputArray.markers.clear();
  for (int index(0); index < output_count; index++)
  {
    visualization_msgs::Marker marker;
    marker.pose.position.x = local_object_mem[index].x();
    marker.pose.position.y = local_object_mem[index].y();
    marker.pose.position.z = 0.0;

    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    marker.color = color;

    marker.id = index;
    marker.header = header;
    marker.action = marker.ADD;
    marker.type = marker.SPHERE;
    outputArray.markers.push_back(marker);
  }

  free(local_object_mem);
  return outputArray;
}

};  // namespace example_gpu_pf



