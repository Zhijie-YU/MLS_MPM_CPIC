import taichi as ti
import numpy as np
import os
import sys

ti.init(arch=ti.gpu) # Try to run on GPU
quality = 1 # Use a larger value for higher-res simulations
n_particles, n_grid = 10000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2 # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters
x = ti.Vector.field(2, dtype=float, shape=n_particles) # position
v = ti.Vector.field(2, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid)) # grid node mass
energy = ti.field(dtype=float, shape=())

gravity = 20
# velocity of the rigid surface
r_v = 1

# num of rigid segments
n_rseg = 100
# location of nodes on the rigid surface
x_r = ti.Vector.field(2, dtype=float, shape=n_rseg+1)
# location of rigid particles
x_rp = ti.Vector.field(2, dtype=float, shape=n_rseg)
x_ls = [0.8, 0.5]
x_le = [1.2, 0.6]

grid_d = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_A = ti.field(dtype=int, shape=(n_grid, n_grid))
grid_T = ti.field(dtype=int, shape=(n_grid, n_grid))

p_d = ti.field(dtype=float, shape=n_particles)
p_A = ti.field(dtype=int, shape=n_particles)
p_T = ti.field(dtype=int, shape=n_particles)
p_n = ti.Vector.field(2, dtype=float, shape=n_particles)

@ti.kernel
def substep():
  line = ti.Vector([x_le[0]-x_ls[0],x_le[1]-x_ls[1]]).normalized()
  for p in x_r:
    x_r[p] = x_r[p] - line * dt * r_v
  for p in x_rp:
    x_rp[p] = x_rp[p] - line * dt * r_v
  
  # CDF
  for i,j in grid_A:
    grid_A[i,j] = 0
    grid_T[i,j] = 0
    grid_d[i,j] = 0.0
  
  for p in x_rp:
    ba = x_r[p+1] - x_r[p]
    base = (x_rp[p] * inv_dx - 0.5).cast(int)
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      pa = (offset + base).cast(float) * dx - x_r[p]
      h = pa.dot(ba) / (ba.dot(ba))

      if h <= 1 and h >= 0:
        grid_d[base + offset] = (pa - h * ba).norm()
        grid_A[base + offset] = 1
        outer = pa[0] * ba[1] - pa[1] * ba[0]

        if outer > 0:
          grid_T[base + offset] = 1
        else:
          grid_T[base + offset] = -1
  
  for p in x:
    p_A[p] = 0
    p_T[p] = 0
    p_d[p] = 0.0
    
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    Tpr = 0.0
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
      offset = ti.Vector([i, j])
      if grid_A[base + offset] == 1:
        p_A[p] = 1
      
      weight = w[i][0] * w[j][1]
      Tpr += weight * grid_d[base + offset] * grid_T[base + offset]
    if p_A[p] == 1:
      if Tpr > 0:
        p_T[p] = 1
      else:
        p_T[p] = -1
      p_d[p] = abs(Tpr)
  
  for i, j in grid_m:
    grid_v[i, j] = [0, 0]
    grid_m[i, j] = 0
  
  # P2G
  for p in x: # Particle state update and scatter to grid (P2G)
    # p is a scalar
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    # Quadratic kernels 
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p] # deformation gradient update
    h = 0.5
    mu, la = mu_0 * h, lambda_0 * h
    U, sig, V = ti.svd(F[p])
    J = 1.0
    for d in ti.static(range(2)):
      new_sig = sig[d, d]
      J *= new_sig
    stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    affine = stress + p_mass * C[p]
    for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood      
      offset = ti.Vector([i, j])
      if p_T[p] * grid_T[base + offset] == -1:
        pass
      else:
        dpos = (offset.cast(float) - fx) * dx
        weight = w[i][0] * w[j][1]
        grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
        grid_m[base + offset] += weight * p_mass
      
    
    
  # grid operation  
  for i, j in grid_m:
    if grid_m[i, j] > 0: # No need for epsilon here
      grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j] # Momentum to velocity
      grid_v[i, j][1] -= dt * gravity # gravity
      
      # seperate boundary      
      if i < 3 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0 # Boundary conditions
      if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
      if j < 3 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
      if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
  
  # G2P
  for p in x: # grid to particle (G2P)
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
    new_v = ti.Vector.zero(float, 2)
    new_C = ti.Matrix.zero(float, 2, 2)
    for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
      g_v = ti.Vector([0.0, 0.0])
      if p_T[p] * grid_T[base + ti.Vector([i,j])] == -1:
        line = ti.Vector([x_le[0]-x_ls[0],x_le[1]-x_ls[1]]).normalized()
        pa = ti.Vector([x[p][0]-x_ls[0], x[p][1]-x_ls[1]])
        np = (pa - pa.dot(line) * line).normalized()
        sg = v[p].dot(np)
        if sg > 0:
          g_v = v[p]
        else:
          g_v = v[p].dot(line) * line
        
      else:
        g_v = grid_v[base + ti.Vector([i, j])]
      
      dpos = ti.Vector([i, j]).cast(float) - fx
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
    v[p], C[p] = new_v, new_C
    x[p] += dt * v[p] # advection

@ti.kernel
def initialize():
  for i in range(n_particles):
    x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.7 + 0.3]
    v[i] = ti.Matrix([0, 0])
    F[i] = ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1
  
  
  x_r[0] = x_ls
  for i in range(n_rseg):
    x_r[i+1] = [x_ls[0] + (x_le[0]-x_ls[0]) / n_rseg * (i+1), x_ls[1] + (x_le[1]-x_ls[1]) / n_rseg * (i+1)]
    x_rp[i] = (x_r[i] + x_r[i+1]) / 2
  
  
initialize()
gui = ti.GUI("Taichi MLS-MPM-Cutting", res=512, background_color=0x112F41)
#video_manager = ti.VideoManager(output_dir="pic/",framerate=24,automatic_build=False)
frame = 0

flag = 1
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
  for s in range(int(5e-3 // dt)):
    substep()

  gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
  gui.line(x_r.to_numpy()[0], x_r.to_numpy()[-1], radius=2, color=0xFF0000)
  
  '''
  grid_A_ = grid_A.to_numpy()
  grid_T_ = grid_T.to_numpy()
  for i in range(grid_A_.shape[0]):
    for j in range(grid_A_.shape[1]):
      if grid_T_[i, j] == 1:
        gui.circle(np.array([i,j])*dx, radius=2, color=0xED553B)
      if grid_T_[i, j] == -1:
        gui.circle(np.array([i,j])*dx, radius=2, color=0xEEEEF0)
  
  '''
  x_ = x.to_numpy()
  p_A_ = p_A.to_numpy()
  p_T_ = p_T.to_numpy()
  for i in range(p_A_.shape[0]):
    if p_T_[i] == 1:
      gui.circle(x_[i], radius=2, color=0xCD00CD)
    if p_T_[i] == -1:
      gui.circle(x_[i], radius=2, color=0x436EEE)
  
  
  #filename = f'pic_cutSolid/frame_{frame:05d}.png'
  #gui.show(filename) 
  gui.show()
  frame += 1
  
