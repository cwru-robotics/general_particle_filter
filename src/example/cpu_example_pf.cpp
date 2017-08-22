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

#include <ros/ros.h>
#include "general_particle_filter/example/cpu_example_pf.h"

namespace example_cpu_pf
{

PlanarParticleFilter::PlanarParticleFilter(int particle_count, const objectVariance &object_variance) :
    cpu_pf::ParticleFilter(particle_count, particle_count),
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

  temp_particles_ = new ObjectState[temp_count];
  output_particles_ = new ObjectState[output_count];
}

void PlanarParticleFilter::deallocateStates()
{
  delete [] temp_particles_;
  delete [] output_particles_;
}

void PlanarParticleFilter::applyAction(const objectAction &object_action)
{
  int temp_count(this->getWeightsSize());

  for (int index(0); index < temp_count; index++)
  {
    output_particles_[index].act(object_action, this->object_variance_, temp_particles_[index]);
  }
}

void PlanarParticleFilter::applyObservation(const ObjectState &object_observation)
{
  int temp_count(this->getWeightsSize());

  for (int index(0); index < temp_count; index++)
  {
    double weight = computeParticleWeight(temp_particles_[index], object_observation, object_variance_);
    this->setParticleWeight(index, weight);
  }
  this->construct_weight_cdf();
  this->sampleParticles();
  for (int index(0); index < getSamplingSize(); index++)
  {
    unsigned int sample_index = getSampleIndex(index);
    output_particles_[index] = temp_particles_[sample_index];
  }
}

ObjectState PlanarParticleFilter::estimateState()
{
  int output_count_i(this->getSamplingSize());
  double output_count_d(static_cast<double>(output_count_i));

  double x(0.0), y(0.0);
  for (int index(0); index < output_count_i; index++)
  {
    if (output_particles_[index].x() < -1000 || output_particles_[index].x() > 1000)
    {
      ROS_INFO("What is going on?");
    }
    x += output_particles_[index].x()/output_count_d;
    y += output_particles_[index].y()/output_count_d;
  }
  return ObjectState(x, y);
}

visualization_msgs::MarkerArray PlanarParticleFilter::getParticleArray(const std_msgs::Header &header,
  const std_msgs::ColorRGBA &color)
{
  visualization_msgs::MarkerArray outputArray;
  outputArray.markers.clear();
  int output_count(this->getSamplingSize());
  for (int index(0); index < output_count; index++)
  {
    visualization_msgs::Marker marker;
    marker.pose.position.x = output_particles_[index].x();
    marker.pose.position.y = output_particles_[index].y();
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
  return outputArray;
}

};  // namespace example_cpu_pf
