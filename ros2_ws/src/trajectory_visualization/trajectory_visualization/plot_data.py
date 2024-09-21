import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sns.set_theme(style="dark")
sns.set_context("paper")

# Load data
data_path = './data/'
plots_path = './plots/'

##########################################################
# I've kept the plots seperately for each environment, 
# so that it's easier to modify individual ones if needed
##########################################################

######################################################################################
################################### Astrobee data ###################################
baseline_pos = np.load(data_path + "particle/baseline_pos.npy")
equivariant_pos = np.load(data_path + "particle/equivariant_pos.npy")
reference_pos = np.load(data_path + "particle/ref_pos.npy")

### Position plots ###
fig, axs = plt.subplots(3, 2)
plot_time = 1000
axs[0, 0].plot(range(plot_time), reference_pos[:plot_time, 0],range(plot_time), baseline_pos[:plot_time, 0])
axs[0, 0].set_ylabel("X (m)")
axs[0, 0].grid(True)
axs[1, 0].plot(range(plot_time), reference_pos[:plot_time, 1],range(plot_time), baseline_pos[:plot_time, 1])
axs[1, 0].set_ylabel("Y (m)")
axs[1, 0].grid(True)
axs[2, 0].plot(range(plot_time), reference_pos[:plot_time, 2],range(plot_time), baseline_pos[:plot_time, 2])
axs[2, 0].set_ylabel("Z (m)")
axs[2, 0].grid(True)

axs[0, 1].plot(range(plot_time), reference_pos[:plot_time, 0],range(plot_time), equivariant_pos[:plot_time, 0])
axs[0, 1].grid(True)
axs[1, 1].plot(range(plot_time), reference_pos[:plot_time, 1],range(plot_time), equivariant_pos[:plot_time, 1])
axs[1, 1].grid(True)
axs[2, 1].plot(range(plot_time), reference_pos[:plot_time, 2],range(plot_time), equivariant_pos[:plot_time, 2])
axs[2, 1].grid(True)

plt.suptitle('Particle Position \n Baseline (left) vs Equivariant (right)')
fig.supxlabel('Timesteps')
plt.savefig(plots_path + 'particle/particle_pos.png', dpi=1000)


######################################################################################
################################### Astrobee data ###################################
baseline_pos = np.load(data_path + "astrobee/baseline_pos.npy")
equivariant_pos = np.load(data_path + "astrobee/equivariant_pos.npy")

baseline_rotm = np.load(data_path + "astrobee/baseline_rotm.npy")
equivariant_rotm = np.load(data_path + "astrobee/equivariant_rotm.npy")

reference_pos = np.load(data_path + "astrobee/ref_pos.npy")
reference_rotm = np.load(data_path + "astrobee/ref_rotm.npy")

reference_eul = R.from_quat(reference_rotm).as_euler('zyx', degrees=True)
baseline_eul = R.from_quat(baseline_rotm).as_euler('zyx', degrees=True)
equivariant_eul = R.from_quat(equivariant_rotm).as_euler('zyx', degrees=True)

### Position plots ###
fig, axs = plt.subplots(3, 2)
plot_time = 1000
axs[0, 0].plot(range(plot_time), reference_pos[:plot_time, 0],range(plot_time), baseline_pos[:plot_time, 0])
axs[0, 0].set_ylabel("X (m)")
axs[0, 0].grid(True)
axs[1, 0].plot(range(plot_time), reference_pos[:plot_time, 1],range(plot_time), baseline_pos[:plot_time, 1])
axs[1, 0].set_ylabel("Y (m)")
axs[1, 0].grid(True)
axs[2, 0].plot(range(plot_time), reference_pos[:plot_time, 2],range(plot_time), baseline_pos[:plot_time, 2])
axs[2, 0].set_ylabel("Z (m)")
axs[2, 0].grid(True)

axs[0, 1].plot(range(plot_time), reference_pos[:plot_time, 0],range(plot_time), equivariant_pos[:plot_time, 0])
axs[0, 1].grid(True)
axs[1, 1].plot(range(plot_time), reference_pos[:plot_time, 1],range(plot_time), equivariant_pos[:plot_time, 1])
axs[1, 1].grid(True)
axs[2, 1].plot(range(plot_time), reference_pos[:plot_time, 2],range(plot_time), equivariant_pos[:plot_time, 2])
axs[2, 1].grid(True)

plt.suptitle('Astrobee Position \n Baseline (left) vs Equivariant (right)')
fig.supxlabel('Timesteps')
plt.savefig(plots_path + 'astrobee/astrobee_pos.png', dpi=1000)

### Attitude plots ###
fig, axs = plt.subplots(3, 2)
plot_time = 1000
axs[0, 0].plot(range(plot_time), reference_eul[:plot_time, 0],range(plot_time), baseline_eul[:plot_time, 0])
axs[0, 0].set_ylabel("Yaw (deg)")
axs[0, 0].grid(True)
axs[1, 0].plot(range(plot_time), reference_eul[:plot_time, 1],range(plot_time), baseline_eul[:plot_time, 1])
axs[1, 0].set_ylabel("Pitch (deg)")
axs[1, 0].grid(True)
axs[2, 0].plot(range(plot_time), reference_eul[:plot_time, 2],range(plot_time), baseline_eul[:plot_time, 2])
axs[2, 0].set_ylabel("Roll (deg)")
axs[2, 0].grid(True)

axs[0, 1].plot(range(plot_time), reference_eul[:plot_time, 0],range(plot_time), equivariant_eul[:plot_time, 0])
axs[0, 1].grid(True)
axs[1, 1].plot(range(plot_time), reference_eul[:plot_time, 1],range(plot_time), equivariant_eul[:plot_time, 1])
axs[1, 1].grid(True)
axs[2, 1].plot(range(plot_time), reference_eul[:plot_time, 2],range(plot_time), equivariant_eul[:plot_time, 2])
axs[2, 1].grid(True)

plt.suptitle('Astrobee Euler angles \n Baseline (left) vs Equivariant (right)')
fig.supxlabel('Timesteps')
plt.savefig(plots_path + 'astrobee/astrobee_eul.png', dpi=1000)

######################################################################################
################################### Quadrotor data ###################################
baseline_pos = np.load(data_path + "quadrotor/baseline_pos.npy")
equivariant_pos = np.load(data_path + "quadrotor/equivariant_pos.npy")

baseline_rotm = np.load(data_path + "quadrotor/baseline_rotm.npy")
equivariant_rotm = np.load(data_path + "quadrotor/equivariant_rotm.npy")

reference_pos = np.load(data_path + "quadrotor/ref_pos.npy")
reference_rotm = np.load(data_path + "quadrotor/ref_rotm.npy")

#print(baseline_pos.shape, equivariant_rotm.shape, reference_pos.shape)

reference_eul = R.from_quat(reference_rotm).as_euler('zyx', degrees=True)
baseline_eul = R.from_quat(baseline_rotm).as_euler('zyx', degrees=True)
equivariant_eul = R.from_quat(equivariant_rotm).as_euler('zyx', degrees=True)



### Position plots ###
fig, axs = plt.subplots(3, 2)
plot_time = 1000
axs[0, 0].plot(range(plot_time), reference_pos[:plot_time, 0],range(plot_time), baseline_pos[:plot_time, 0])
axs[0, 0].set_ylabel("X (m)")
axs[0, 0].grid(True)
axs[1, 0].plot(range(plot_time), reference_pos[:plot_time, 1],range(plot_time), baseline_pos[:plot_time, 1])
axs[1, 0].set_ylabel("Y (m)")
axs[1, 0].grid(True)
axs[2, 0].plot(range(plot_time), reference_pos[:plot_time, 2],range(plot_time), baseline_pos[:plot_time, 2])
axs[2, 0].set_ylabel("Z (m)")
axs[2, 0].grid(True)

#fig, axs = plt.subplots(3, 1)
axs[0, 1].plot(range(plot_time), reference_pos[:plot_time, 0],range(plot_time), equivariant_pos[:plot_time, 0])
axs[0, 1].grid(True)
axs[1, 1].plot(range(plot_time), reference_pos[:plot_time, 1],range(plot_time), equivariant_pos[:plot_time, 1])
axs[1, 1].grid(True)
axs[2, 1].plot(range(plot_time), reference_pos[:plot_time, 2],range(plot_time), equivariant_pos[:plot_time, 2])
axs[2, 1].grid(True)

plt.suptitle('Quadrotor Position \n Baseline (left) vs Equivariant (right)')
fig.supxlabel('Timesteps')
plt.savefig(plots_path + 'quadrotor/quadrotor_pos.png', dpi=1000)

### Attitude plots ###
fig, axs = plt.subplots(3, 2)
plot_time = 1000
axs[0, 0].plot(range(plot_time), reference_eul[:plot_time, 0],range(plot_time), baseline_eul[:plot_time, 0])
axs[0, 0].set_ylabel("Yaw (deg)")
axs[0, 0].grid(True)
axs[1, 0].plot(range(plot_time), reference_eul[:plot_time, 1],range(plot_time), baseline_eul[:plot_time, 1])
axs[1, 0].set_ylabel("Pitch (deg)")
axs[1, 0].grid(True)
axs[2, 0].plot(range(plot_time), reference_eul[:plot_time, 2],range(plot_time), baseline_eul[:plot_time, 2])
axs[2, 0].set_ylabel("Roll (deg)")
axs[2, 0].grid(True)

#fig, axs = plt.subplots(3, 1)
axs[0, 1].plot(range(plot_time), reference_eul[:plot_time, 0],range(plot_time), equivariant_eul[:plot_time, 0])
axs[0, 1].grid(True)
axs[1, 1].plot(range(plot_time), reference_eul[:plot_time, 1],range(plot_time), equivariant_eul[:plot_time, 1])
axs[1, 1].grid(True)
axs[2, 1].plot(range(plot_time), reference_eul[:plot_time, 2],range(plot_time), equivariant_eul[:plot_time, 2])
axs[2, 1].grid(True)

plt.suptitle('Quadrotor Euler angles \n Baseline (left) vs Equivariant (right)')
fig.supxlabel('Timesteps')
plt.savefig(plots_path + 'quadrotor/quadrotor_eul.png', dpi=1000)
######################################################################################