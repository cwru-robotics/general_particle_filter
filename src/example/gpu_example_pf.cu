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
  int padded_temp_count(this->getPaddedWeightsSize());
  int padded_output_count(this->getPaddedSamplingSize());

  ObjectState* temp_particles = new ObjectState[padded_temp_count];
  ObjectState* output_particles = new ObjectState[padded_output_count];

  printf("Allocating %d temp particles and %d output particles.\n", padded_output_count, padded_temp_count);

  // allocate the gpu mem and perform the copy operator... (not sure what the best way is)
  cudaError_t  success_01(cudaMalloc(&d_temp_particles_, sizeof(ObjectState) * padded_temp_count));
  cudaError_t  success_02(cudaMalloc(&d_output_particles_, sizeof(ObjectState) * padded_output_count));

  printf("allocated particles: %s, %s\n", cudaGetErrorString(success_01), cudaGetErrorString(success_02));

  success_01 = cudaMalloc(&d_obj_action_, sizeof(objectAction));
  success_02 = cudaMalloc(&d_obj_var_,   sizeof(objectVariance));

  printf("allocated IO: %s, %s\n", cudaGetErrorString(success_01), cudaGetErrorString(success_02));

  objectAction temp_action(0.0, 0.0);

  cudaMemcpy(d_obj_action_, &temp_action, sizeof(objectAction), cudaMemcpyHostToDevice);
  cudaError_t err = cudaGetLastError();
  printf("copy 1 temp particles: %s\n", cudaGetErrorString(err));
  cudaMemcpy(d_obj_var_, &object_variance_, sizeof(objectVariance), cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  printf("copy 2 temp particles: %s\n", cudaGetErrorString(err));

  cudaMemcpy(d_temp_particles_, temp_particles, sizeof(ObjectState) * padded_temp_count, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  printf("copy 3 temp particles: %s\n", cudaGetErrorString(err));

  cudaMemcpy(d_output_particles_, output_particles, sizeof(ObjectState) * padded_output_count, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  printf("copy 4 temp particles: %s\n", cudaGetErrorString(err));

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
  int padded_temp_count(this->getWeightsSize());
  int output_count(this->getSamplingSize());
  int padded_output_count(this->getPaddedSamplingSize());

  double * d_rand;
  cudaMalloc(&d_rand, temp_count * 2 * sizeof(double));
  curandGenerator_t prngGPU;
  curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
  std::random_device rd;

  curandSetPseudoRandomGeneratorSeed(prngGPU, rd());

  curandGenerateNormalDouble(prngGPU, d_rand, padded_temp_count * 2, 0.0, 1.0);

  cudaMemcpy(d_obj_action_, &object_action, sizeof(objectAction), cudaMemcpyHostToDevice);

  uint blockCount = padded_output_count / THREADBLOCK_SIZE;

  // printf("calling the action kernel <%d, %d>\n", blockCount, THREADBLOCK_SIZE);
  applyActionKernel<<<blockCount, THREADBLOCK_SIZE>>>(d_obj_action_,
    reinterpret_cast<double2*> (d_rand), d_temp_particles_, d_output_particles_, d_obj_var_, temp_count);

  cudaFree(d_rand);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("free error: %s\n", cudaGetErrorString(err));

}

void PlanarParticleFilter::applyObservation(const ObjectState &object_observation)
{
  int padded_output_count(this->getPaddedSamplingSize());
  int output_count(this->getSamplingSize());
  uint blockCount = padded_output_count / THREADBLOCK_SIZE;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("pre Compute particle weight error: %s\n", cudaGetErrorString(err));

  computeParticleWeights<<<blockCount, THREADBLOCK_SIZE>>>(d_temp_particles_, d_weights_pdf_,
      object_observation, object_variance_, output_count);

  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Compute particle weight error: %s\n", cudaGetErrorString(err));

  this->construct_weight_cdf();
  this->sampleParticleIndices();
  copyParticlesByIndex<<<blockCount, THREADBLOCK_SIZE>>>(d_output_particles_,
    this->d_sample_indices_, d_temp_particles_, output_count);

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



