# general_particle_filter
[![Build Status](https://travis-ci.com/cwru-robotics/general_particle_filter.svg?token=YmHMxBbcdppbMMkZWTut&branch=master)](https://travis-ci.com/cwru-robotics/general_particle_filter)

This repository defines a package that provides the framework for implementing a particle filter either on a CPU or a GPU.
Please note that the cuda support is contigent on using cmake 3.9 or later which is not shipped with ubuntu xenial (as of 1/2018).
To get a more upto date version of cmake, please go to the [cmake download page](https://cmake.org/download/). 
Once the archive is downloaded and extracted, it can be installed using the previous verion of cmake.
```
mkdir build
cd build
cmake ..
make
sudo make install
```

## CPU Based Particle Filter

### CPU demo

To run this demo use the following commands:

```bash
roslaunch general_particle_filter example_cpu.launch
```

This demo launches both a simple planar simulator as well as an rviz interface where the simulator can be watched.
The time per filter loop execution is displayed.

## GPU (CUDA) Based Particle Filter and Demo

### GPU (CUDA) demo

```bash
roslaunch general_particle_filter example_gpu.launch
```

This demo launches both a simple planar simulator as well as an rviz interface where the simulator can be watched.
The time per filter loop execution is displayed.