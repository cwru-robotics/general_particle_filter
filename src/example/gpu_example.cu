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



#include <iostream>
#include <cuda.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PolygonStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <general_particle_filter/gpu/particle_filter.h>
#include <general_particle_filter/example/gpu_example_pf.h>


// construct a header (to use with all message generation)
std_msgs::Header gen_header(unsigned int index)
{
  std_msgs::Header results;
  results.seq = index;
  results.frame_id = "map";
  results.stamp = ros::Time::now();

  return results;
}

geometry_msgs::PolygonStamped generatePath(const std_msgs::Header &header)
{
  geometry_msgs::PolygonStamped circle;
  circle.header = header;
  circle.polygon.points.resize(200);
  for (int index = 0; index < 200; index++)
  {
    double angle(static_cast<double>(index) * M_PI * 2/200.0);

    circle.polygon.points[index].x = 8.0 * cos(angle);
    circle.polygon.points[index].y = 8.0 * sin(angle);
    circle.polygon.points[index].z = 0.0;
  }
  return circle;
}

int main(int argc, char** argv)
{
  // intialize the ros node
  ros::init(argc, argv, "pf_ex_CPU");
  ros::NodeHandle nh;

  // construct the publishers: (no subscribers)
  // @TODO publish marker array of states
  // @TODO publish goal motion?
  ros::Publisher action_pub(nh.advertise<visualization_msgs::Marker>("planar_actions", 1));
  ros::Publisher state_pub(nh.advertise<geometry_msgs::PointStamped>("state", 1));
  ros::Publisher state_obs_pub(nh.advertise<geometry_msgs::PointStamped>("obs_state", 1));
  ros::Publisher state_est_pub(nh.advertise<geometry_msgs::PointStamped>("est_state", 1));
  ros::Publisher particle_pub(nh.advertise<visualization_msgs::MarkerArray>("particle_list", 1));
  ros::Publisher path_pub(nh.advertise<geometry_msgs::PolygonStamped>("target_path", 1));

  int particle_count(50);
  ROS_INFO("Starting the example PF with %d particles", particle_count);

  example_gpu_pf::objectVariance objectVariance_act(0.1, 0.1, 0.2, 0.2);
  example_gpu_pf::objectVariance objectVariance_pf(0.15, 0.15, 0.3, 0.3);

  example_gpu_pf::PlanarParticleFilter planar_pf(particle_count, objectVariance_pf);

  example_gpu_pf::ObjectState act_state;
  example_gpu_pf::ObjectState est_state(act_state.observe(objectVariance_act));
  ROS_INFO("The initial observed position is <%3.3f, %3.3f>", est_state.x(), est_state.y());

  example_gpu_pf::ObjectState initial_state(planar_pf.estimateState());
  ROS_INFO("The initial particle position is <%3.3f, %3.3f>", initial_state.x(), initial_state.y());

  // green for actual and action
  std_msgs::ColorRGBA act_color;
  act_color.a = 1.0;
  act_color.b = 0.1;
  act_color.g = 1.0;
  act_color.r = 0.1;

  // blue for estimate
  std_msgs::ColorRGBA est_color;
  act_color.a = 1.0;
  act_color.b = 1.0;
  act_color.g = 0.1;
  act_color.r = 0.1;

  // yellow for observation
  std_msgs::ColorRGBA obs_color;
  act_color.a = 1.0;
  act_color.b = 0.1;
  act_color.g = 1.0;
  act_color.r = 1.0;

  // faded cyan for particles
  std_msgs::ColorRGBA pf_color;
  pf_color.a = 0.5;
  pf_color.b = 1.0;
  pf_color.g = 1.0;
  pf_color.r = 0.1;

  unsigned int index(0);
  while (ros::ok())
  {
    // get a new header
    std_msgs::Header header(gen_header(index));
    example_gpu_pf::objectAction obj_act(example_gpu_pf::controlLaw(est_state));

    ROS_INFO("The action is <%3.3f, %3.3f>", obj_act.dx_, obj_act.dy_);

    action_pub.publish(action_marker(obj_act, act_state, header, act_color));

    example_gpu_pf::ObjectState new_state = act_state.act(obj_act, objectVariance_act);
    act_state = new_state;

    example_gpu_pf::ObjectState obs_state(act_state.observe(objectVariance_act));

    state_obs_pub.publish(obs_state.statePoint(header));

    // @TODO time this part
    int start_s = clock();
    planar_pf.applyAction(obj_act);
    planar_pf.applyObservation(obs_state);
    est_state = planar_pf.estimateState();
    visualization_msgs::MarkerArray particle_array(planar_pf.getParticleArray(header, pf_color));
    int stop_s = clock();
    double dt((stop_s-start_s)/ static_cast<double>(CLOCKS_PER_SEC)*1000);
    ROS_INFO("The filter took %4.4f seconds to process %d particles", dt, particle_count);
    ROS_INFO("The aggregated position is <%3.3f, %3.3f>", est_state.x(), est_state.y());
    ROS_INFO("The actual position is <%3.3f, %3.3f>", act_state.x(), act_state.y());

    particle_pub.publish(particle_array);
    state_pub.publish(act_state.statePoint(header));
    state_est_pub.publish(est_state.statePoint(header));

    geometry_msgs::PolygonStamped new_path(generatePath(header));
    path_pub.publish(new_path);

    index++;
    ros::Duration(0.75).sleep();
  }
  return 0;
}
