#!/usr/bin/env python
import rospy
import sys
import tf
import pudb

import sys
import numpy as np
from threading import Thread, RLock

import angles


import random


from copy import deepcopy
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from car_sim.msg import State, Action
from swarm_sim.msg import Intruder
import car_sim.car_manifold_model_python as man_py

import observer_model as observer
import object_tracker.car_state_to_marker as convert
from object_tracker.intruder_model import intruder_odds
import car_sim.random_car_generator as random_car


class particle_filter(object):
    """
    Represents a particle filter object

    This particle filter utilizes a state and observer class to keep track of
        particles, update particles, and sample particles accordingly

    Attributes:
        motion_model            : How the particle behaves in the world
        policy                  : The state based on the current action
        transition_probability  : The probability, given position and the last
                                    compramisation, that I am now compromised
        copy_state              : Function to copy the state #var.deepcopy could
                                    be used
        residual_pdf            : Used to determine pdf of the distance between
                                    two particles (Accounts for noise R)
        h_fn                    : Used to obtain a mapping from a state space to
                                    sensor reading (Accounts for noise Q)
        point_list              : List of all the state point particles
        result_points           : Temporary list of particles as result of
                                    resampling
        comp_list               : List of compromised status of each particle
        weight_list             : List of the weights for each particle

    Constants:
        MULTIPLIER : The amount of time the particles will be duplicated in
                        resample. Also the amount to multiply by in order to
                        obtain the ACTUAL amount of particles used
        particle_count : The size of the particle list (will be multiplied by
                        multiplier)
    """
    def __init__(self, car_name_, car_intrinsics_, particle_count_, intruder_multiplier_):
        """
        Initialize particle filter

        Args:
            state_func : Used to obtain motion model,
                zones (transition), and state manipulation
            obs_func : Used to obtain sesory data transformation
            init_guess : Optional Paramater to insert a best guess into the
                initialization
        """
        self.car_name = car_name_
        self.car_intrinsics = car_intrinsics_
        self.particle_count = particle_count_

        self.car_particles = [State()] * self.particle_count

        self.action = Action()
        self.action_queue = 0

        self.last_time = None

        self.lock = RLock()

        self.initialized = False

        self.control_sub = None
        self.state_sub = None
        self.particle_pub = None

        self.intruder_multiplier = intruder_multiplier_

        self.total_particles = (1 + self.intruder_multiplier) * self.particle_count

        self.temp_car_particles = [State()] * self.particle_count * (1 + self.intruder_multiplier)
        self.particle_weights = np.zeros(self.particle_count * (1 + self.intruder_multiplier))

        self.comp_rate = -1

    def start_ros_rviz_interface(self):
        """
        Starts running the publishers and subscribers required for interfacing to rviz and the car simulator.
        """
        self.last_time = rospy.get_rostime()

        self.control_sub = rospy.Subscriber(self.car_name + "/cmd_vel", Action, self.action_callback)
        self.state_sub = rospy.Subscriber(self.car_name + "/full_state", State,
                                          self.state_callback)
        self.particle_pub = rospy.Publisher(self.car_name + "/particle_pose", MarkerArray,
                                            queue_size=1)

        # self.temp_particle_pub = rospy.Publisher(self.car_name + "/temp_particle_pose", MarkerArray,
        #                                     queue_size=1)

        self.comp_rate_pub = rospy.Publisher(self.car_name + "/comp_rate", Float64,
                                             queue_size=1)

    def prb_weight_prtcl(self, sensor_point, last_prob):
        """
        Returns the weight for a specific particle

        Used to obtain the weight for a particle given a sensor point and the
            last compromised state

        Args:
            sensor_point: Incoming reading from sensor data
            last_prob: The last computed probability to be used in the weight

        Returns:
            The product of the transitional probability and pdf of the residual
                function

        """
        state_in_obs = self.h_fn(self.comp_list[i])
        trns_prb = self.transition_probability(state_in_obs, sensor_point)
        diff_Q = self.residual_pdf(state_in_obs, sensor_point)
        return last_prob * trns_prb * diff_Q

    def normalize_weights(self, tot_weight):
        """Normalizes weights accross particles

        Takes a total weight and normalizes weights from the object weight_list

        Args:
            tot_weight: The total weight of the particles in the weight_list

        """
        for idx in range(self.total_particles):
            self.particle_weights[idx] = self.particle_weights[idx] / tot_weight

    def sample_points(self):
        """Samples points based on weights

        Based on a normalized list of weights, an update is computed via
        sampling. 1/MULTIPLIER particles are samples and as such the
        sampled particles are multiplied accross the compromisation space.

        """
        sample_interval = float(1)/self.particle_count
        offset = random.random() * sample_interval

        last_weight = self.particle_weights[0]
        index_weight = 0

        new_car_particles = [State()] * self.particle_count
        comp_rate_num = 0.0
        comp_rate_den = 0.0

        # sample higher weighted particles into result_points
        for i in range(self.particle_count):
            current_weight = offset + i * sample_interval

            while current_weight > last_weight:
                index_weight += 1
                last_weight = self.particle_weights[index_weight]

            new_car_particles[i] = deepcopy(self.temp_car_particles[index_weight])

            if self.temp_car_particles[index_weight].intruderStatus.intruder != Intruder.OK:
                comp_rate_num += 1.0
            comp_rate_den += 1.0

        self.comp_rate = comp_rate_num / comp_rate_den

        self.car_particles[:] = new_car_particles[:]

    def action_callback(self, action_):
        """
        updates points in particle list through a motion model

        Args:
            action: The published action.

        """
        with self.lock:
            self.action = action_

    def state_callback(self, state_):
        """
            Updates the particles based on a new observation of the car_object.
        Args:
            self: The particle filter object
            state: the incoming object state
        """
        print self.action.header.seq
        print state_.header.seq
        print "matching"

        """if self.initialized == False :
            self.initialized = True

            with self.lock:
                for idx in range(self.particle_count):

                    new_state = deepcopy(state_)
                    new_state.pose.x = state_.pose.x + random.uniform(-1, 1)
                    new_state.pose.y = state_.pose.y + random.uniform(-1, 1)
                    new_state.pose.theta = angles.normalize_angle(state_.pose.theta + random.uniform(-0.5, 0.5))
                    self.car_particles[idx] = new_state

        else:
            now = rospy.get_rostime()
            rosdt = now - self.last_time
            dt = rosdt.to_sec()
            set_time = True

            with self.lock:
                # Feed forward estimation of the particle evolution
                tot_weight = 0.0

                base_index = -1

                for index in range(0, self.total_particles):
                    # compute the new overal index:
                    if index % self.particle_count == 0:
                        base_index += 1

                    compStatus = (self.car_particles[base_index].intruderStatus.intruder != Intruder.OK)
                    if index % self.particle_count == 0:
                        # The car transitions to not compromised:
                        pre_weight = (1 - intruder_odds(compStatus))
                        actionTemp = deepcopy(self.action)
                        nextIntruder = Intruder.OK
                    else:
                        # The car transitions to comprised:
                        actionTemp = random_car.generate_random_car_action(self.car_intrinsics)
                        pre_weight = (intruder_odds(compStatus)) / self.intruder_multiplier
                        nextIntruder = Intruder.CONTROLLER

                    new_car_particle = man_py.car_kinematics(self.car_particles[base_index],
                                                         actionTemp, self.car_intrinsics, dt)

                    new_car_particle.intruderStatus.intruder = nextIntruder
                    new_car_particle.velocity = state_.velocity
                    self.temp_car_particles[index] = new_car_particle

                    new_weight = observer.observation_probability(new_car_particle, state_, self.car_intrinsics) \
                                 * pre_weight
                    self.particle_weights[index] = tot_weight + new_weight
                    tot_weight = self.particle_weights[index]

                # resampling
                #print tot_weight
                #print self.particle_weights
                self.normalize_weights(tot_weight)
                #print self.particle_weights

                #self.pub_unsampled_markers(now)
                self.sample_points()

                self.last_time = now

            # why not
            # self.pub_unsampled_markers(now)

            # publish the markers:
            self.pub_markers(now)

            # publish the overall compromisation:
            self.pub_comp()"""

    def pub_markers(self, time):
        """
        Publishes the tracked particles as a list of markers
        """
        frame_id = "map"
        uncomp_rbg = [.1, .1, 1]
        comp_rbg = [1, .1, .1]

        particle_markers = MarkerArray()

        # adds points in point list to publishable marker array
        for idx, car_particle in enumerate(self.car_particles):

            marker = Marker()
            marker.pose = convert.car_state_to_pose(car_particle)

            marker.scale.x = 10.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0

            marker.color.a = 1.0
            if car_particle.intruderStatus.intruder > 0:
                marker.color.r, marker.color.b, marker.color.g = comp_rbg
            else:
                marker.color.r, marker.color.b, marker.color.g = uncomp_rbg

            marker.id = idx
            marker.header.frame_id = frame_id
            marker.header.stamp = time
            marker.action = marker.ADD

            particle_markers.markers.append(marker)

        self.particle_pub.publish(particle_markers)

    def pub_comp(self):
        """
        publishes the weighted compromisation of the particle filter:
        """
        self.comp_rate_pub.publish(self.comp_rate)
