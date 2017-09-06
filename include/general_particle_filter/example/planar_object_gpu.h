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

#ifndef GENERAL_PARTICLE_FILTER_EXAMPLE_PLANAR_OBJECT_GPU_H
#define GENERAL_PARTICLE_FILTER_EXAMPLE_PLANAR_OBJECT_GPU_H

#include <random>
#include <cuda_runtime.h>
#include <cuda.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>

namespace example_gpu_pf
{




// @TODO Only implement operators and supporting functions AS NEEDED!

/**
 * @brief The object action.
 */
struct objectAction
{
  /**
   * @brief The X-Y components of the action.
   */
  double dx_;
  double dy_;

  /**
   * @brief This is the default constructor of the ObjectAction struct.
   *
   * @param dx
   * @param dy
   */
  explicit __device__ __host__ objectAction(double dx = 0.0, double dy = 0.0):
    dx_(dx), dy_(dy)
  {
  };

  /**
   * @brief This is the copy constructor (used for function calls.)
   *
   * @param obj The object being copied
   */
  __device__ __host__ objectAction(const objectAction& obj):
  dx_(obj.dx_),
  dy_(obj.dy_)
  {
  };

};

/**
 * @brief the internal object variance.
 *
 * Both process and observation variance.
 */
struct objectVariance
{
  /**
   * @brief The std dev in action, b_x and b_y.
   */
  double b_x_;
  double b_y_;

  /**
   * @brief The std in observation c_x and c_y
   */
  double c_x_;
  double c_y_;

  /**
   * @brief The copy constructor.
   * @param object_variance
   */
  __device__ __host__ objectVariance(const objectVariance& object_variance):
    b_x_(object_variance.b_x_),
    b_y_(object_variance.b_y_),
    c_x_(object_variance.c_x_),
    c_y_(object_variance.c_y_)
  {
  };

  /**
   * @brief The elementwise constructor of the variance.
   *
   * @param b_x
   * @param b_y
   * @param c_x
   * @param c_y
   */
  explicit __device__ __host__ objectVariance(double b_x = 0.0, double b_y = 0.0, double c_x = 0.0, double c_y = 0.0):
    b_x_(b_x), b_y_(b_y), c_x_(c_x), c_y_(c_y)
  {
  };
};

/**
 * @brief The Object State itself.
 *
 * These states are the particle used in the particle filter.
 */
class ObjectState
{
private:
  /**
   * @brief, the X and Y position of the object.
   */
  double x_pose_;
  double y_pose_;

  /**
   * @brief the workspace limits of the object state.
   */
  const static int x_limit_g = 10;
  const static int y_limit_g = 10;

public:
  /**
   * @brief The explicit constructor (also serves as a default)
   *
   * @param x_pose
   * @param y_pose
   */
  explicit __device__ __host__ ObjectState(double x_pose = 0.0, double y_pose = 0.0): x_pose_(x_pose), y_pose_(y_pose)
  {
  }

  /**
   * @brief The assignment operator.
   *
   * @param object_state
   * @return a reference to the resulting object
   */
   __device__ __host__ ObjectState& operator = (const ObjectState& object_state);


  /**
   * @brief generate an observation of the object.
   * @return The observation of the objects state.
   */
  __host__ ObjectState observe(const objectVariance& obj_var);

  /**
   * @brief generate a new Object state from the car's action
   *
   * @param obj_act
   * @param obj_var
   * @return The new object
   */
  __host__ ObjectState act(const objectAction& obj_act, const objectVariance& obj_var);

  /**
   * @brief Applies the action to a state object and stores it in a provided pointer.
   *
   * @param obj_act The action
   * @param obj_var The process variance container
   * @param obj_ref The pointer to the object for storage
   */
  __host__ __device__ void act(const objectAction &obj_act, const double* random_scale,
    const objectVariance &obj_var, ObjectState &obj_ref);

  __device__ void act_dev(objectAction *obj_act, double* random_scale,
                               objectVariance *obj_var, ObjectState *obj_ptr);


  /**
   * @brief clamps the object position so that it stays inside the limits of the workspace
   */
  __device__ __host__ void clampPose();

  /**
   * @brief Get the x position of the object.
   *
   * @return The x-position
   */
  __device__ __host__ double x() const
  {
    return x_pose_;
  }

  /**
   * @brief Get the y position of the object.
   *
   * @return The y-position
   */
  __device__ __host__ double y() const
  {
    return y_pose_;
  }

  /**
   * @brief get a stamped point for publishing and displaying in RVIZ.
   * @param header The message header
   * @return the message to publish.
   */
  __host__ geometry_msgs::PointStamped statePoint(const std_msgs::Header &header);
};

/**
 * @brief This function converts the action (&state) to a Marker for publication
 *
 * @param action The action of the object
 * @param state The objects current state
 * @param header The header of the topic.
 * @param color the marker color.
 * @return The Marker for publishing
 */
visualization_msgs::Marker action_marker(const objectAction &action, const ObjectState& state,
  const std_msgs::Header &header, const std_msgs::ColorRGBA& color);

/**
 * @brief This is the object control law for circle following
 *
 * The control law moves the object about a circle (counter clock wise)
 * The law will also push the object towards the circle as it moves around.
 *
 * @param state The state of the object
 *
 * @return The computed action
 */
__host__ objectAction controlLaw(const ObjectState& state);

/**
 * @brief Estimates the weight of the observation and prediction
 *
 * @param prediction
 * @param observation
 * @param obj_var
 *
 * @return The estimated weight.
 */
__device__ __host__ double computeParticleWeight(const ObjectState& prediction,
  const ObjectState& observation, const objectVariance& obj_var);

__global__ void applyActionKernel(example_gpu_pf::objectAction* obj_act, double *randomList,
                                  example_gpu_pf::ObjectState* obj_input, example_gpu_pf::ObjectState* obj_result,
                                  example_gpu_pf::objectVariance* obj_var);


__global__ void computeParticleWeights(example_gpu_pf::ObjectState* object_particles,
                                       double * weights, example_gpu_pf::ObjectState object_observation,
                                       objectVariance object_variance);

}  // namespace example_gpu_pf

#endif  // GENERAL_PARTICLE_FILTER_EXAMPLE_PLANAR_OBJECT_GPU_H
