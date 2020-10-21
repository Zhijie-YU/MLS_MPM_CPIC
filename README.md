# MLS_MPM_CPIC
## Introduction
CPIC(Compatible Particle-in-Cell) cases implemented with Taichi lang where MLS-MPM and rigid body collision techniques are adopted.  
These cases mainly based on the paper [A Moving Least Squares Material Point Method with Displacement
Discontinuity and Two-Way Rigid Body Coupling](https://github.com/Zhijie-YU/MLS_MPM_CPIC/blob/main/2018_mlsmpm.pdf).  
For running codes, just install Python and corresponding Taichi package.  
For installing Taichi, please refer to [Taichi Doc](https://taichi.readthedocs.io/en/stable/) for details.

## Cutting Thin
In this case, a suspended solid block is cut by a thin plate.  
No penalty force and impulse on the rigid body are implemented.  

![image](https://github.com/Zhijie-YU/MLS_MPM_CPIC/blob/main/figures/cut.gif)

## Rotating Fan
In this case, a rotating fan interacts with granular particles flowing down.  
The number of fan blades and its initial angular velocity can be adpated. 

![image](https://github.com/Zhijie-YU/MLS_MPM_CPIC/blob/main/figures/fan.gif)

