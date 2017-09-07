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
#include <general_particle_filter/example/planar_object_gpu.h>

namespace example_gpu_pf
{


ObjectState& ObjectState::operator = (const ObjectState& object_state)
{
  this->x_pose_ = object_state.x_pose_;
  this->y_pose_ = object_state.y_pose_;

  return *this;
}

ObjectState ObjectState::observe(const objectVariance& obj_var)
{
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());
  std::normal_distribution<double> rand_norm(0.0, 1.0);

  double d_x(rand_norm(gen) * obj_var.c_x_);
  double d_y(rand_norm(gen) * obj_var.c_y_);

  return ObjectState(this->x_pose_ + d_x, this->y_pose_ + d_y);
}

ObjectState ObjectState::act(const objectAction& obj_act, const objectVariance &obj_var)
{
  std::mt19937 gen;
  std::random_device rd;
  gen.seed(rd());
  std::normal_distribution<double> rand_norm(0.0, 1.0);

  double *random_d = new double[2];
  random_d[0] = rand_norm(gen);
  random_d[1] = rand_norm(gen);

  ObjectState resultState;
  this->act(obj_act, random_d, obj_var, resultState);
  return resultState;
}

void ObjectState::act(const objectAction &obj_act, const double* random_d,
  const objectVariance &obj_var, ObjectState &obj_ref)
{
  double d_x(random_d[0] * obj_var.b_x_ * obj_act.dx_  + obj_act.dx_);
  double d_y(random_d[1] * obj_var.b_y_ * obj_act.dy_ + obj_act.dy_);
  obj_ref.x_pose_ = this->x_pose_ + d_x;
  obj_ref.y_pose_ = this->y_pose_ + d_y;
  obj_ref.clampPose();
}

__device__ void ObjectState::act(objectAction *obj_act, double2* random_d,
                                     objectVariance *obj_var, ObjectState *obj_ptr)
{
  double d_x(random_d->x * obj_var->b_x_ * obj_act->dx_ + obj_act->dx_);
  double d_y(random_d->y * obj_var->b_y_ * obj_act->dy_ + obj_act->dy_);
  obj_ptr->x_pose_ = this->x_pose_ + d_x;
  obj_ptr->y_pose_ = this->y_pose_ + d_y;
  obj_ptr->clampPose();
}

__global__ void applyActionKernel(objectAction* obj_act, double2 *randomList,
                                  ObjectState* obj_result, ObjectState* obj_input,
                                  objectVariance* obj_var, uint particle_count)
{
  unsigned int idx =  blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < particle_count)
  {
    obj_input[idx].act(obj_act, randomList + idx, obj_var, obj_result + idx);
  }
}


__global__ void copyParticlesByIndex(example_gpu_pf::ObjectState* d_dst, uint* d_src_index,
  example_gpu_pf::ObjectState* d_src, uint particle_count)
{
  unsigned int idx =  blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < particle_count)
  {
    d_dst[idx] = d_src[d_src_index[idx]];
  }
}

__global__ void computeParticleWeights(example_gpu_pf::ObjectState* object_particles, double * weights,
  example_gpu_pf::ObjectState object_observation,
  objectVariance object_variance, uint particle_count)
{
  unsigned int idx =  blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < particle_count)
  {
    weights[idx] = computeParticleWeight(object_particles[idx], object_observation, object_variance);
  }
}

void ObjectState::clampPose()
{
  if (this->x_pose_ > x_limit_g)
  {
    this->x_pose_ = static_cast<double> (x_limit_g);
  }
  else
  {
    if (this->x_pose_ < -x_limit_g)
    {
      this->x_pose_ = static_cast<double> (-x_limit_g);
    }
  }

  if (this->y_pose_ > y_limit_g)
  {
    this->y_pose_ = static_cast<double> (y_limit_g);
  }
  else
  {
    if (this->y_pose_ < -y_limit_g)
    {
      this->y_pose_ = static_cast<double> (-y_limit_g);
    }
  }
}

geometry_msgs::PointStamped ObjectState::statePoint(const std_msgs::Header &header)
{
  geometry_msgs::PointStamped result;

  result.header = header;
  result.point.x = this->x();
  result.point.y = this->y();
  result.point.z = 0.0;

  return result;
}



visualization_msgs::Marker action_marker(const objectAction &action, const ObjectState& state,
  const std_msgs::Header &header, const std_msgs::ColorRGBA& color)
{
  visualization_msgs::Marker results;

  results.header = header;
  results.pose.position.x = state.x();
  results.pose.position.y = state.y();
  results.pose.position.z = 0.0;

  double angle = atan2(action.dy_, action.dx_);
  results.pose.orientation.x = 0.0;
  results.pose.orientation.y = 0.0;
  results.pose.orientation.z = sin(angle/2);
  results.pose.orientation.w = cos(angle/2);

  results.scale.x = 0.5;
  results.scale.y = 0.25;
  results.scale.z = 1.0;

  results.color = color;
  results.action = results.ADD;
  results.type = results.ARROW;

  return results;
}

objectAction controlLaw(const ObjectState& state)
{
  objectAction results;

  double des_rad(8.0);
  double act_rad(sqrt(state.y()*state.y() + state.x()*state.x()));
  double rad_error(des_rad - act_rad);
  if (act_rad < 0.1)
  {
    results.dx_ = 0.9 * rad_error;
    results.dy_ = 0.0;
  }
  else
  {
    results.dx_ = 0.9 * rad_error * state.x() / act_rad - 0.0125 * state.y();
    results.dy_ = 0.9 * rad_error * state.y() / act_rad + 0.0125 * state.x();
  }
  return results;
}

double computeParticleWeight(const ObjectState& prediction,
  const ObjectState& observation, const objectVariance& obj_var)
{
  double x_error(prediction.x() - observation.x());
  double y_error(prediction.y() - observation.y());

  double factor(exp(-x_error*x_error/(obj_var.c_x_*obj_var.c_x_) - y_error*y_error/(obj_var.c_y_*obj_var.c_y_)));
  return factor;
}

};  // namespace example_gpu_pf
