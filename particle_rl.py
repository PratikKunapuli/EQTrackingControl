#################################################
############### Import Libraries ################
import tqdm
import torch
import numpy as np
import scipy.signal
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from torch.distributions.normal import Normal
################################################

# Device
device = "cpu"

class particle:

    def __init__(self, x0 = np.random.rand(6), xd = np.zeros(3,), xd_dot = np.zeros(3,), xd_ddot = np.zeros(3,)):

        self.dt = 0.05
        self.N = 2000
        self.x = x0

        self.xd = xd
        self.xd_dot = xd_dot
        self.xd_ddot = xd_ddot

        self.t = 0

        self.T_SCALE = 2.0
        
    def step(self, a):

        self.t += self.dt

        ###########################################
        ############ Trajectory tracking ##########
        ###########################################


        ###### Lissajous, with time scaling ####### 
        # self.xd[0] = np.sin(self.t / self.T_SCALE)
        # self.xd[1] = -0.5 * np.sin(2 * self.t  / self.T_SCALE)
        # self.xd[2] = 1.0

        # self.xd_dot[0] = np.cos(self.t / self.T_SCALE) / self.T_SCALE
        # self.xd_dot[1] = -np.cos(2 * self.t / self.T_SCALE) / self.T_SCALE
        # self.xd_dot[2] = 0.0

        # self.xd_ddot[0] = -np.sin(self.t / self.T_SCALE) / (self.T_SCALE ** 2)
        # self.xd_ddot[1] = 2 * np.sin(2 * self.t / self.T_SCALE)  / (self.T_SCALE ** 2)
        # self.xd_ddot[2] = 0.0

        # self.x[3:] = self.x[3:] + a * self.dt
        # self.x[:3] = self.x[:3] + self.x[3:] * self.dt

        #Huber Loss-type reward
        #term1 = particle.huber(self.x[:3], self.xd)
        #term2 = particle.huber(self.x[3:], self.xd_dot)
        #reward = -0.01 * (term1 + term2)

        #MSE Loss-type reward
        reward = -0.01 * (np.linalg.norm(self.xd - self.x[:3]) ** 2 + np.linalg.norm(self.xd_dot - self.x[3:]) ** 2)

        ###### Stabilize at a point #####
        # self.xd[0] = 2.0
        # self.xd[1] = 2.0
        # self.xd[2] = 2.0

        # self.xd_dot[0] = 0
        # self.xd_dot[1] = 0
        # self.xd_dot[2] = 0

        # self.xd_ddot[0] = 0
        # self.xd_ddot[1] = 0
        # self.xd_ddot[2] = 0

        self.x[3:] = self.x[3:] + a * self.dt
        self.x[:3] = self.x[:3] + self.x[3:] * self.dt

        # reward = -0.01 * (np.linalg.norm(self.xd - self.x[:3]) ** 2 + np.linalg.norm(self.xd_dot - self.x[3:]) ** 2)

        return reward

    def reset(self, x0 = np.random.rand(6), xd = np.zeros(3,), xd_dot = np.zeros(3,), xd_ddot = np.zeros(3,)):

        self.x = x0

        self.xd = xd
        self.xd_dot = xd_dot
        self.xd_ddot = xd_ddot

        self.t = 0

    @staticmethod
    def huber(x, xd, delta=1.0):

        abs_e = np.abs(x - xd)
        quad = np.minimum(abs_e, delta)
        linear = (abs_e - quad)
        
        return np.mean(0.5 * quad ** 2 + delta * linear)

##################################################################################################################
############################################  ESE 6500: BIPEDAL MJPC #############################################
###########################################        CUSTOM PPO      ###############################################
########################################### ADAPTED FROM SPINNING UP #############################################
##################################################################################################################

###############################################################################################################################
################################################ Buffer #######################################################################
# Create a Buffer class to store and process trajectories
class DataBuffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0,0,size
    
    def store(self, obs, act, rew, val, logp):

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]

        self.ret_buf[path_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma)], rews[::-1], axis=0)[::-1][:-1]

        self.path_start_idx = self.ptr

    def get(self):

        assert self.ptr == self.max_size

        self.ptr, self.path_start_idx = 0,0
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()

        # Add 1e-4 to avoid small denominators
        self.adv_buf = (self.adv_buf - adv_mean)/(adv_std + 1e-4)

        data = dict(obs = self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}
#################################################################################################################################



###########################################################################################
############################# Stochastic Actor Network ####################################
class MLPActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_dim):

        super().__init__()

        # Keep std high initialliy to encourage exploration
        self.log_std = torch.nn.Parameter((-0.35 * torch.ones(act_dim)).to(device))

        self.mu_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim,), nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim,), nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim,), nn.LeakyReLU(),
            nn.Linear(hidden_dim,hidden_dim,), nn.LeakyReLU(),
            nn.Linear(hidden_dim, act_dim), nn.Identity(),
            #nn.Linear(hidden_dim, act_dim), nn.Tanh(),
        ).to(device)

    def _distribution(self, obs):
        mu = self.mu_net(obs.to(device))
        std = torch.exp(self.log_std).to(device)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1).to(device)

    def forward(self, obs, act=None):

        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        
        return pi, logp_a
############################################################################################

############################################################################################
############################### Value function (Critic) Network ############################
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes,):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes,), nn.LeakyReLU(),
            nn.Linear(hidden_sizes,hidden_sizes,), nn.LeakyReLU(),
            nn.Linear(hidden_sizes,hidden_sizes,), nn.LeakyReLU(),
            nn.Linear(hidden_sizes,hidden_sizes,), nn.LeakyReLU(),
            nn.Linear(hidden_sizes, 1), nn.Identity(),
        ).to(device)

    def forward(self, obs):
        return torch.squeeze(self.value_net(obs), -1).to(device)
############################################################################################


###############################################################################
############################# PPO Network #####################################
class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=128,):

        super().__init__()

        self.pi = MLPActor(obs_dim, action_dim, hidden_dim).to(device)
        self.v = MLPCritic(obs_dim, hidden_dim).to(device)

    def step(self, obs):

        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)

        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
##############################################################################


##########################################################################################
##############################  PPO Training #############################################
def PPO(env_fn, actor_critic=MLPActorCritic, xdim=12, udim=3, hidden_dim=32,):

    # Tuned Parameters 
    '''
    1) Early stopping     - Train pi and vf for only max 10 gradient descent steps
    2) Reduce Clip ratio  - Keep the controllers very very close to each other
    3) Less learning rate - Prevent overfitting as pi and vf networks are relatively powerful
    '''
    steps_per_epoch = 500
    epochs = 5000
    gamma = 0.99
    clip_ratio = 0.12
    pi_lr = 1e-3
    vf_lr = 1e-3
    train_pi_iters = 5
    train_v_iters = 5
    lam = 0.97
    max_ep_len = 2000
    target_kl = 0.01

    # Random seed
    torch.manual_seed(10)
    np.random.seed(10)

    # Create the PPO network
    ac = actor_critic(obs_dim=xdim, action_dim=udim, hidden_dim=hidden_dim).to(device)

    # Create a buffer to store all the info
    buf = DataBuffer(obs_dim=xdim, act_dim=udim, size=steps_per_epoch, gamma=gamma, lam=lam)

    # pi loss function
    def compute_loss_pi(obs, act, adv, logp_old):

        _, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old.to(device))
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        # Define a loss coeff to delay loss
        loss_coeff = 1.0
        loss_pi = -(loss_coeff * torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_old - logp).mean().item()

        return loss_pi, approx_kl

    # Vf loss function
    def compute_loss_v(obs, ret):
        
        return ((ac.v(obs) - ret)**2).mean()
    
    # Define optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Update function called after trajectory ends
    def update(epoch):
        data = buf.get()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, kl = compute_loss_pi(data['obs'], data['act'], data['adv'], data['logp'])
            if kl > 1.0 * target_kl:
                # Early stopping. KL divergence diverging too much - break"
                break
            loss_pi.backward()
            pi_optimizer.step()

        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data['obs'], data['ret'])
            loss_v.backward()
            vf_optimizer.step()

    # START TRAINING 
    # env_data, ep_ret, ep_len = env_fn.reset(), 0, 0
    # x = env_data.observation
    # o = np.array(x['orientations'].tolist() + [x['height']] + x['velocity'].tolist())

    ep_ret, ep_len = 0, 0
    x0 = np.random.multivariate_normal(np.zeros(6),
                                       #np.diag([1, 1, 1, 0.1, 0.1, 0.1]),
                                       0.5 * np.eye(6),)
    
    xd_set = np.random.multivariate_normal(np.zeros(3), 3 * np.eye(3))

    # xd_set = np.array([
    #     [2, 2, 2],
    #     [-2, -2, 1.1],
    #     [-1.7, 2.1, -2],
    #     [2, 0.8, -0.7],
    #     #[5, -5, -5],
    #     #[-5, -5, -5],
    #     ],
    # )

    # xd_set = np.array([
    #     [8, 6, 4],
    #     [-5, -8, -3],
    #     [2.6, 7.4, 6.8],
    #     [-8.1, 7.6, -7.3],
    # ])

    #xd = np.random.multivariate_normal(10 + np.zeros(3,), np.diag([1, 1, 0.1]))

    env_fn = particle(x0, xd=xd_set,)
    o = np.concatenate((env_fn.x, env_fn.xd,))

    total_timesteps = 0

    log_data = []

    print("Training for " + str(epochs) + "epochs!")

    for epoch in tqdm.tqdm(range(epochs)):
        ret = 0
        ep_ret, ep_len = 0, 0

        #x0 = np.random.multivariate_normal(np.zeros(6,), np.diag([1, 1, 1, 0.1, 0.1, 0.1]),)
        x0 = np.random.multivariate_normal(np.zeros(6,), 
                                       #np.diag([1, 1, 1, 0.1, 0.1, 0.1]),
                                       0.5 * np.eye(6))

        xd_set = np.random.multivariate_normal(np.zeros(3), 3 * np.eye(3))

        env_fn = particle(x0=x0, xd=xd_set,)

        o = np.concatenate((env_fn.x, env_fn.xd))


        for t in range(steps_per_epoch):

            a, v, logp = ac.step(torch.from_numpy(o).float().unsqueeze(0).to(device))

            r = env_fn.step(a)
            next_o = np.concatenate((env_fn.x, env_fn.xd))

            ret += r

            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v.item(), logp.item())

            o = next_o.copy()

            timeout = ep_len == max_ep_len
            ### TERMINATE if the walker falls on the ground!!!
            terminal = timeout or np.any(np.abs(env_fn.x[:3]) > 10)
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.from_numpy(o).float().unsqueeze(0).to(device))
                else:
                    v = 0

                ep_ret, ep_len = 0, 0
                #x0 = np.random.multivariate_normal(np.zeros(6,), np.diag([1, 1, 1, 0.1, 0.1, 0.1]),)

                # xd_set = np.array([
                #     [5, 5, 5],
                #     [-5, 5, 5],
                #     [5, -5, -5],
                #     [-5, -5, -5]]
                # )
                x0 = np.random.multivariate_normal(np.zeros(6,), 
                                       #np.diag([1, 1, 1, 0.1, 0.1, 0.1]),
                                       0.5 * np.eye(6))
                #xd = np.random.multivariate_normal(10 + np.zeros(3,), np.diag([1, 1, 0.1]))

                xd_set = np.random.multivariate_normal(np.zeros(3), 3 * np.eye(3))

                env_fn = particle(x0, xd=xd_set,)
                o = np.concatenate((env_fn.x, env_fn.xd,))

                buf.finish_path(v)

        update(epoch)

        total_timesteps += t

        log_data.append([t, total_timesteps, ret, 11111] + [round(entry.item(), 2) for entry in ac.pi.log_std])

        np.savetxt("./trained_models/non_eq/train_data/data.txt", np.array(log_data))

        # Save every 50 epochs
        if epoch % 50 == 0:
            torch.save(ac.state_dict(), "./trained_models/non_eq/train_data/model_" + str(epoch) + ".pth")
            #print(total_timesteps, ret)

    print("===========   Training Over ... =============")

    torch.save(ac.state_dict(), "./trained_models/non_eq/train_data/final_model.pth")
#############################################################################################################################

### EXTRACT ENVELOPES ###
'''
Taken from: https://stackoverflow.com/questions/34235530/how-to-get-high-and-low-envelope-of-a-signal
'''
def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax


###########################  TRAIN WALKER USING PPO! ######################################
if __name__ == "__main__":
    """
    Setup walker environment
    """
    e = particle()
    udim=3
    xdim=9

    # Chose what to do
    train_model = True
    play_trained_model = True

    # Train the PPO network
    if train_model: 
        PPO(env_fn=e, actor_critic=MLPActorCritic, xdim=xdim, udim=udim, )

    # See how the model trained
    if play_trained_model:

        data_path = "./trained_models/non_eq/train_data/data.txt"
        model_path = "./trained_models/non_eq/train_data/final_model.pth"

        # Load the trained model
        ac = MLPActorCritic(obs_dim=xdim, action_dim=udim, hidden_dim=32).to(device)
        ac.load_state_dict(torch.load(model_path))

        # x0 = np.random.multivariate_normal(np.zeros(6,), 
        #                                    #np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        #                                    np.eye(6),
        #                                    )

        # xd_set = np.array([
        #     [2, 2, 2],
        #     [-2, -2, 1.1],
        #     [-1.7, 2.1, -2],
        #     [2, 0.8, -0.7],
        # ])

        xd_set = np.random.multivariate_normal(np.zeros(3), 3 * np.eye(3))

        # xd_set = np.array([
        # [8, 6, 4],
        # [-5, -8, -3],
        # [2.6, 7.4, 6.8],
        # [-8.1, 7.6, -7.3],
        # #[6.4, -2.7, 6.3],
        # ])

        #e = particle(x0, xd=xd_set[np.random.randint(xd_set.shape[0]), :],)

        N = 10 * e.N

        x_xd_history_set = []

        for _ in range(25):

            x0 = np.random.multivariate_normal(np.zeros(6,), 
                                            #np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                                            0.5 * np.eye(6),
                                        )

            xd_set = np.random.multivariate_normal(np.zeros(3), 3 * np.eye(3))

            e = particle(x0, xd=xd_set,)

            x_history = np.zeros((N, 3))
            x_dot_history = np.zeros((N, 3))

            xd_history = np.zeros((N, 3))
            xd_dot_history = np.zeros((N, 3))
            a_history = np.zeros((N, 3))

            for i in range(N):

                obs = np.concatenate((e.x, e.xd))

                a, _, _ = ac.step(torch.from_numpy(obs).float().unsqueeze(0).to(device))

                e.step(a)

                if np.any(np.abs(e.x[:3]) > 10): 
                    print("AAAAAHHHHH")
                    break
                    

                x_history[i] = e.x[:3]
                x_dot_history[i] = e.x[3:]

                xd_history[i] = e.xd
                a_history[i] = a

            x_xd_history_set.append((x_history, xd_history))

        # Plot the trajectory
        ax = plt.figure(100).add_subplot(projection="3d")

        #ax.scatter(xd_set[:, 0], xd_set[:, 1], xd_set[:, 2], "*")

        for x_xd in x_xd_history_set:
            ax.plot(x_xd[0][:, 0], x_xd[0][:, 1], x_xd[0][:, 2], linewidth=0.5)
            ax.scatter(x_xd[1][-1, 0], x_xd[1][-1, 1], x_xd[1][-1, 2], s = 0.75)

        ax.grid(True)
        ax.axis("equal")
        ax.legend(r"$x(t)$", r"$x_d(t)$")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Particle Trajectory")
        plt.savefig("./trained_models/non_eq/outputs/traj.png", dpi=900)
        plt.show()

        plt.figure(101)

        plt.plot(a_history[:, 0])
        plt.plot(a_history[:, 1])
        plt.plot(a_history[:, 2])

        plt.grid(True)
        plt.title("Policy")
        plt.savefig("./trained_models/non_eq/outputs/policy.png")

        # Plot the return
        raw_return_data = np.loadtxt(data_path)[:, 2][:5000]

        # Smoothen return plot
        kernel_size = 50
        kernel = np.ones(kernel_size) / kernel_size
        smooth_return_data = np.convolve(raw_return_data, kernel, mode="same")

        lmin, lmax = hl_envelopes_idx(raw_return_data, dmin=25, dmax=25)
        xaxis = np.arange(0, len(raw_return_data))

        fig = plt.figure(1)

        fig.tight_layout()
        ax = fig.gca()
        ax.grid(which = "major", linewidth = 1, alpha=1.)
        ax.grid(which = "minor", linewidth = 0.5, alpha=0.5)
        ax.minorticks_on()
        ax.plot(xaxis[:4900], smooth_return_data[:4900], c="red")
        #print(len(lmin), len(lmax))
        min_len = min(len(lmin), len(lmax))
        ax.fill_between(xaxis[lmin[:min_len]], raw_return_data[lmin[:min_len]], raw_return_data[lmax[:min_len]], facecolor="red", linewidth=0.5, alpha=0.3, edgecolor=None)

        ax.set_ylim([-195, 5])
        ax.set_xlabel("Rollouts")
        ax.set_ylabel("Returns")
        ax.set_title(r"Statistics of Return $G_t$")
        plt.savefig("./trained_models/non_eq/outputs/return.png")

        # Plot variation of log_std
        log_std = np.loadtxt(data_path)[:, 4:]
        fig = plt.figure(2)

        fig.tight_layout()

        ax = fig.gca()
        ax.grid(which = "major", linewidth = 1, alpha=1.)
        ax.grid(which = "minor", linewidth = 0.5, alpha=0.5)
        ax.minorticks_on()
        ax.plot(log_std[:, 0])
        ax.plot(log_std[:, 1])
        ax.plot(log_std[:, 2])

        ax.legend(["Action 1", "Action 2", "Action 3"])

        ax.set_xlabel("Rollouts")
        ax.set_ylabel(r"$log(\sigma_a)$")
        ax.set_title(r"Variation of $log(\sigma_a)$ during training")
        plt.savefig("./trained_models/non_eq/outputs/log_sigma.png")

###########################################################################################