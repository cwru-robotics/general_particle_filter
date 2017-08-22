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

#ifndef GENERAL_PARTICLE_FILTER_EXAMPLE_CPU_EXAMPLE_PF_H
#define GENERAL_PARTICLE_FILTER_EXAMPLE_CPU_EXAMPLE_PF_H

#include <std_msgs/Header.h>
#include <visualization_msgs/MarkerArray.h>


#include <general_particle_filter/gpu/particle_filter.h>
#include <general_particle_filter/example/planar_object_gpu.h>

namespace example_gpu_pf
{

/**
 * @brief This is an example implementation of a Planar Particle Filter.
 */
class PlanarParticleFilter: private gpu_pf::ParticleFilter
{
public:
  /**
   * @brief The explicit constructor
   *
   * @param particle_count The number of particles to use.
   * @param object_variance The internal object variance.
   */
  explicit PlanarParticleFilter(int particle_count, const objectVariance& object_variance);

  /**
   * @brief The explicit default destructor
   */
  ~PlanarParticleFilter();

  /**
   * @brief Applies the specified action to the particle list.
   *
   * @param object_action
   */
  void applyAction(const objectAction& object_action);

  /**
   * @brief Applies the specified observation to the particle list.
   *
   * Resampling is done here.
   *
   * @param object_observation
   */
  void applyObservation(const ObjectState &object_observation);

  /**
   * @brief Estimates the aggregate state of the object.
   *
   * In this instance, the aggregated state is achieved through averaging.
   *
   * @return The estimated object state.
   */
  ObjectState estimateState();

  /**
   * @brief Gets and array of markers for publishing to RVIZ
   *
   * @param header The message header to use
   * @param color  The marker array colors
   *
   * @return The marker array
   */
  visualization_msgs::MarkerArray getParticleArray(const std_msgs::Header &header,
    const std_msgs::ColorRGBA &color);

private:
  /**
   * @brief A stored copy of the local object variance.
   */
  objectVariance object_variance_;

  /**
   * The objectState pointers.
   *
   * Temporary (before resampling)
   * output (after resampling)
   */
  ObjectState* d_temp_particles_;
  ObjectState* d_output_particles_;

  /**
   * @brief Pointers for the executed action and object variance.
   */
  objectAction* d_obj_action_;
  objectVariance* d_obj_var_;

  /**
   * @brief allocates the object state memory
   */
  void allocateStates();

  /**
   * @brief deallocates the object state memory.
   */
  void deallocateStates();
};

}  // namespace example_cpu_pf

#endif  // GENERAL_PARTICLE_FILTER_EXAMPLE_CPU_EXAMPLE_PF_H
