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
kh = 5 # penalty stiffness parameter
dy = 0.1 # dynamic friction coefficient for rigid surface
x = ti.Vector.field(2, dtype=float, shape=n_particles) # position
v = ti.Vector.field(2, dtype=float, shape=n_particles) # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles) # deformation gradient
material = ti.field(dtype=int, shape=n_particles) # material id
Jp = ti.field(dtype=float, shape=n_particles) # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid)) # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid)) # grid node mass


gravity = 10

n_bodies = 5
# num of rigid segments
n_rseg = 100
# location of nodes on the rigid surface
x_r = ti.Vector.field(2, dtype=float, shape=(n_rseg+1,n_bodies))
# location of rigid particles
x_rp = ti.Vector.field(2, dtype=float, shape=(n_rseg,n_bodies))
# velocity of the rigid particles
v_rp = ti.Vector.field(2, dtype=float, shape=(n_rseg,n_bodies))

x_ls = ti.Vector.field(2, dtype=float, shape=n_bodies)
x_le = ti.Vector.field(2, dtype=float, shape=())
m_line = ti.field(dtype=float, shape=())
J_line = ti.field(dtype=float, shape=())
Mt = ti.field(dtype=float, shape=()) # angular momentum change
omega = ti.field(dtype=float, shape=()) # angular velocity 

grid_d = ti.Vector.field(n_bodies, dtype=float, shape=(n_grid, n_grid))
grid_A = ti.Vector.field(n_bodies, dtype=int, shape=(n_grid, n_grid))
grid_T = ti.Vector.field(n_bodies, dtype=int, shape=(n_grid, n_grid))
# rigid body index closest to grid node
grid_r = ti.field(dtype=ti.i32, shape=(n_grid, n_grid))
# rigid particle of different bodies index closet to grid node
grid_rp = ti.field(dtype=ti.i32, shape=(n_grid, n_grid, n_bodies))


p_d = ti.Vector.field(n_bodies, dtype=float, shape=n_particles)
p_A = ti.Vector.field(n_bodies, dtype=int, shape=n_particles)
p_T = ti.Vector.field(n_bodies, dtype=int, shape=n_particles)
p_n = ti.Vector.field(2, dtype=float, shape=(n_particles,n_bodies))


@ti.kernel
def substep():
  Mt[None] = 0.0
  # CDF
  for i,j in grid_A:
    for k in ti.static(range(n_bodies)):
      grid_A[i,j][k] = 0
      grid_T[i,j][k] = 0
      grid_d[i,j][k] = -1.0
      grid_rp[i,j,k] = -1
    grid_r[i,j] = -1
  
  for k in ti.static(range(n_bodies)):
    for p in range(n_rseg):
      ba = x_r[p+1,k] - x_r[p,k]
      base = (x_rp[p,k] * inv_dx - 0.5).cast(int)
      for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
        offset = ti.Vector([i, j])
        pa = (offset + base).cast(float) * dx - x_r[p,k]
        h = pa.dot(ba) / (ba.dot(ba))

        if h <= 1 and h >= 0:
          grid_d[base + offset][k] = (pa - h * ba).norm()
          grid_A[base + offset][k] = 1
          temp = base + offset
          grid_rp[temp[0], temp[1], k] = p
          cross = pa[0] * ba[1] - pa[1] * ba[0]
          #print(grid_d[base + offset])
          if cross > 0:
            grid_T[base + offset][k] = 1
          else:
            grid_T[base + offset][k] = -1
    
  for i,j in grid_r:
    for k in ti.static(range(n_bodies)):
      d_min = 0.0
      if grid_A[i,j][k] == 1:
        if grid_r[i,j] == -1:
          d_min = grid_d[i,j][k]
          grid_r[i,j] = k
        else:
          if grid_d[i,j][k] < d_min:
            d_min = grid_d[i,j][k]
            grid_r[i,j] = k
  
  for k in ti.static(range(n_bodies)):
    for p in x:
      p_A[p][k] = 0
      p_T[p][k] = 0
      p_d[p][k] = 0.0
      
      base = (x[p] * inv_dx - 0.5).cast(int)
      fx = x[p] * inv_dx - base.cast(float)
      w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
      Tpr = 0.0
      
      d_vecs = ti.Vector([0,0,0,0,0,0,0,0,0]).cast(float)
      diag = ti.Matrix.identity(float, 9)
      temp = ti.Vector([0,0,0]).cast(float)
      Q = ti.Matrix.rows([temp,temp,temp,temp,temp,temp,temp,temp,temp]) 
      
      for i, j in ti.static(ti.ndrange(3, 3)): # Loop over 3x3 grid node neighborhood
        offset = ti.Vector([i, j])
        if grid_A[base + offset][k] == 1:
          p_A[p][k] = 1
        
        nn = (i+1)*(j+1)-1
        d_sign = grid_d[base + offset][k] * grid_T[base + offset][k] 
        weight = w[i][0] * w[j][1]
        dpos = (offset.cast(float) - fx) * dx
      
        for mn in ti.static(range(9)):
          if mn == nn:
            d_vecs[mn] = d_sign
            diag[mn,mn] = weight
            Q[mn,0] = 1
            Q[mn,1] = dpos[0]
            Q[mn,2] = dpos[1]
            
        Tpr += weight * grid_d[base + offset][k] * grid_T[base + offset][k]
      if p_A[p][k] == 1:
        if p_T[p][k] == 0:
          if Tpr > 0:
            p_T[p][k] = 1
          else:
            p_T[p][k] = -1
        M = Q.transpose() @ diag @ Q
        M = M.inverse()
        dist_p = M @ Q.transpose() @ diag @ d_vecs
        p_d[p][k] = dist_p[0]
        p_n[p,k] = ti.Vector([dist_p[1],dist_p[2]]).normalized()
      else:
        p_T[p][k] = 0     
  
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
    mu = 0 # roughly imitate fluid
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
      flag = 1
      # check compatibility
      for k in ti.static(range(n_bodies)):
        if p_T[p][k] == grid_T[base + offset][k] or p_T[p][k] * grid_T[base + offset][k] == 0:
          pass
        else:
          flag = 0
      if flag:
        # compatible particle and grid node
        dpos = (offset.cast(float) - fx) * dx
        weight = w[i][0] * w[j][1]
        grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
        grid_m[base + offset] += weight * p_mass
          
    
  # grid operation  
  for i, j in grid_m:
    if grid_m[i, j] > 0: # No need for epsilon here
      grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j] # Momentum to velocity
      grid_v[i, j][1] -= dt * gravity # gravity
      
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
      offset = ti.Vector([i,j])
      g_v = ti.Vector([0.0, 0.0])
      flag = 1
      # check compatibility
      for k in ti.static(range(n_bodies)):
        if p_T[p][k] == grid_T[base + offset][k] or p_T[p][k] * grid_T[base + offset][k] == 0:
          pass
        else:
          flag = 0
      if flag == 0:
        r_body = grid_r[base + offset]
        temp = base + offset
        r_id = grid_rp[temp[0], temp[1], r_body]

        line = (x_r[r_id+1,r_body] - x_r[r_id,r_body]).normalized()
        pa = x[p] - x_r[r_id,r_body]
        np = (pa - pa.dot(line) * line).normalized()
        sg = (v[p]-v_rp[r_id,r_body]).dot(np)
        if sg > 0:
          g_v = v[p]
        else:
          vt = (v[p]-v_rp[r_id,r_body]) - sg * np
          xi = max(0, vt.norm()+dy*sg)
          g_v = vt.normalized() * xi + v_rp[r_id,r_body]
          
          # accumulate angular momentum
          rp = x_rp[r_id,r_body] - x_r[n_rseg,r_body]
          weight = w[i][0] * w[j][1] 
          mvp = p_mass * weight * (v[p] - g_v)
          Mt += rp[0] * mvp[1] - rp[1] * mvp[0] # cross product for 2D
        
      else:
        g_v = grid_v[base + offset]
      
      dpos = ti.Vector([i, j]).cast(float) - fx
      weight = w[i][0] * w[j][1]
      new_v += weight * g_v
      new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
    v[p], C[p] = new_v, new_C
    
    # penalty force
    for k in ti.static(range(n_bodies)):
      if p_T[p][k] * p_d[p][k] < 0:
        f_penalty = kh * p_d[p][k] * p_n[p,k]
        v[p] += dt * f_penalty / p_mass
        
    x[p] += dt * v[p] # advection
   
  # rigid body advection
  dw = Mt / J_line
  omega[None] += dw
  
  og_Vec = ti.Vector([0.0,0.0,omega])
  for p,body in x_rp:
    rp = x_rp[p,body] - x_r[n_rseg,body]
    rp_Vec = ti.Vector([rp[0], rp[1], 0.0])
    vrp = og_Vec.cross(rp_Vec)
    #print(vrp)
    v_rp[p,body] = ti.Vector([vrp[0], vrp[1]])
    
    
  for p,body in x_rp:
    x_rp[p,body] = x_rp[p,body] + dt * v_rp[p,body]
  for j in ti.static(range(n_bodies)):
    for i in range(n_rseg-1):
      x_r[i+1,j] = (x_rp[i,j]+x_rp[i+1,j]) / 2.0
    x_r[0,j] = 2 * x_rp[0,j] - x_r[1,j]
    x_r[n_rseg,j] = 2 * x_rp[n_rseg-1,j] - x_r[n_rseg-1,j]
  # better avoid using [-1] as the index

@ti.kernel
def initialize():
  for i in range(n_particles):
    x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.6 + 0.3]
    v[i] = ti.Matrix([0, 0])
    F[i] = ti.Matrix([[1, 0], [0, 1]])
    Jp[i] = 1
  
  x_le[None] = [0.7, 0.15]
  length = 0.1
  pi = 3.1415926
  for i in range(n_bodies):
    x_ls[i] = [x_le[None][0]+length*ti.cos((1+i*2/n_bodies)*pi), x_le[None][1]+length*ti.sin((1+i*2/n_bodies)*pi)]
  
  m_line[None] = 0.0002 # mass for each blade
  J_line[None] = m_line * length**2 / 3.0 * n_bodies # total J for a fan
  omega[None] = -15 # initial angular velocity
  
  for j in ti.static(range(n_bodies)):
    x_r[0,j] = x_ls[j]
    for i in range(n_rseg):
      x_r[i+1,j] = x_ls[j] + (x_le[None]-x_ls[j]) / n_rseg * (i+1)
      x_rp[i,j] = (x_r[i,j] + x_r[i+1,j]) / 2
  
  
  
initialize()
gui = ti.GUI("Taichi MLS-MPM-Fan", res=512, background_color=0x112F41)
#video_manager = ti.VideoManager(output_dir="pic/",framerate=24,automatic_build=False)
frame = 0
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
  for s in range(int(5e-3 // dt)):
    substep()
  gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
  for i in range(n_bodies):
    gui.line(x_r.to_numpy()[0,i], x_r.to_numpy()[-1,i], radius=2.2, color=0xFF0000)
  
  x_ = x.to_numpy()
  p_A_ = p_A.to_numpy()
  p_T_ = p_T.to_numpy()
  for i in range(p_A_.shape[1]):
    for j in range(p_A_.shape[0]):
      if p_T_[j,i] == 1:
        gui.circle(x_[j], radius=1.5, color=0xCD00CD)
      if p_T_[j,i] == -1:
        gui.circle(x_[j], radius=1.5, color=0x436EEE)
  
  
  #filename = f'pic_fanRotation/frame_{frame:05d}.png'
  #gui.show(filename)
  gui.show()
  frame += 1
  
