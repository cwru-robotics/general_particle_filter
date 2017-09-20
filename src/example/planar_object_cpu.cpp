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
#include <general_particle_filter/example/planar_object_cpu.h>

namespace example_cpu_pf
{

std::mt19937 ObjectState::gen_ = std::mt19937(0);

std::normal_distribution<double> ObjectState::rand_norm_ = std::normal_distribution<double>(0.0, 1.0);

void ObjectState::seedRNG(int seed)
{
  if (seed < 0)
  {
    std::random_device rd;
    gen_.seed(rd());
  }
  else
  {
    gen_.seed(static_cast<unsigned int>(seed));
  }
}

ObjectState& ObjectState::operator = (const ObjectState& object_state)
{
  this->x_pose_ = object_state.x_pose_;
  this->y_pose_ = object_state.y_pose_;
}

ObjectState ObjectState::observe(const objectVariance& obj_var)
{
  double d_x(rand_norm_(gen_) * obj_var.c_x_);
  double d_y(rand_norm_(gen_) * obj_var.c_y_);

  return ObjectState(this->x_pose_ + d_x, this->y_pose_ + d_y);
}

ObjectState ObjectState::act(const objectAction& obj_act, const objectVariance &obj_var)
{
  ObjectState resultState;
  this->act(obj_act, obj_var, resultState);
  return resultState;
}

void ObjectState::act(const objectAction &obj_act, const objectVariance &obj_var, ObjectState &obj_ref)
{
  double d_x(rand_norm_(gen_) * obj_var.b_x_ * obj_act.dx_  + obj_act.dx_);
  double d_y(rand_norm_(gen_) * obj_var.b_y_ * obj_act.dy_ + obj_act.dy_);
  if (std::abs(d_x) > 100 || std::abs(d_y) > 100)
  {
    ROS_INFO("Action result is <%4.4f, %4.4f>", d_x, d_y);
  }
  obj_ref.x_pose_ = this->x_pose_ + d_x;
  obj_ref.y_pose_ = this->y_pose_ + d_y;
  obj_ref.clampPose();
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

  results.dx_ = 0.9 * rad_error * state.x()/act_rad - 0.0125 * state.y();
  results.dy_ = 0.9 * rad_error * state.y()/act_rad + 0.0125 * state.x();
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

};  // namespace example_cpu_pf
