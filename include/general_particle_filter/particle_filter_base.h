/*
 *  particle_filter_base.h
 *  Copyright (C) 2017  Russell Jackson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * The particle_filter.h defines an abstract base that can be used as a template for making fully functional particle filters
 */


#ifndef GENERAL_PARTICLE_FILTER_PARTICLE_FILTER_BASE_H
#define GENERAL_PARTICLE_FILTER_PARTICLE_FILTER_BASE_H

namespace pf
{

class ParticleFilterBase
{

  /**
   * @brief This function allocates the necessary arrays for resampling etc.
   *
   * @param the number of particles to allocate for.
   * 
   * @return 0 on success, error code on failure.
   */
  int	allocateMem(int n) = 0;

  void deallcateMem() = 0;

  
};

};  // namespace cpu_pf

#endif  // GENERAL_PARTICLE_FILTER_CPU_PARTICLE_FILTER_H