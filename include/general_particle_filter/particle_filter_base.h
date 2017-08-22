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

/**
 * This file defines the abstract particle filter. The intent is for the specialized CPU and GPU particle filters to
 * derive from this class.
 *
 * TODO identify if class derivation is indeed the best strategy.
 */

#ifndef GENERAL_PARTICLE_FILTER_PARTICLE_FILTER_BASE_H
#define GENERAL_PARTICLE_FILTER_PARTICLE_FILTER_BASE_H

namespace base_pf
{

class ParticleFilterBase
{
  /**
   * @brief This function allocates the necessary arrays for resampling etc.
   *
   * @param n the number of particles to allocate for.
   * 
   * @return 0 on success, error code on failure.
   */
  int allocateMem(int n) = 0;

  /**
   * @brief This function deallocates the particle filter arrays.
   */
  void deallocateMem() = 0;
};

};  // namespace base_pf

#endif  // GENERAL_PARTICLE_FILTER_PARTICLE_FILTER_BASE_H
