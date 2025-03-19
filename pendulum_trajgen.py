import gymnasium as gym
import numpy as np
import sys

# -------------------------
# Settings
# -------------------------
n_traj = 12000#5000     # Number of trajectories to generate
horizon = 400      # Number of steps per trajectory
seed_val= 100
np.random.seed(seed_val)

# Create the Pendulum-v1 environment
env = gym.make("Acrobot-v4")
# env = gym.make("InvertedPendulum-v4")#
# env = gym.make("Pendulum-v5")

# We want to record the state as [theta, theta_dot]
state_dim = env.observation_space.shape[0]#2      # theta and theta_dot
action_dim = env.action_space.shape[0]  # typically 1

# Pre-allocate arrays to store the data:
# Each trajectory has 'horizon' transitions.
# For each transition, we record:
#   - x_t: the current state [theta, theta_dot]
#   - u_t: the control input
#   - x_t+1: the subsequent state [theta, theta_dot]
X = np.zeros((n_traj, horizon, state_dim))      # states x_t
U = np.zeros((n_traj, horizon, action_dim))       # actions u_t
X_next = np.zeros((n_traj, horizon, state_dim))   # next states x_t+1

# -------------------------
# Data Generation Loop
# -------------------------
for traj in range(n_traj):
    # Sample an initial angle uniformly from [-pi, pi] and set angular velocity to zero.
    # Reset the environment and override its internal state.
    obs, info = env.reset()
    qpos = np.array([np.random.uniform(low=-np.pi*(170/180), high=np.pi*(170/180)),np.random.uniform(low=-np.pi*(170/180), high=np.pi*(170/180))])
    qvel = np.array([0,0])#0+np.random.uniform(low=-0.4, high=0.4),0])
    # qpos = np.array([np.pi+np.random.uniform(low=-np.pi*(170/180), high=np.pi*(170/180))])
    # qvel = np.array([0])
    env.unwrapped.set_state(qpos, qvel)
    
    # Instead of using the default observation (which is [cos(theta), sin(theta), theta_dot]),
    # we directly record the underlying state.
    # obs = env.unwrapped.state.copy()  # state = [theta, theta_dot]
    # print(obs)
    k = 0
    flag = True
    for t in range(horizon):
        # Record the current state x_t (as [theta, theta_dot])
        # if t%5==0:
        #     flag = True
        # else:
        #     flag = False
        
        # Sample a random action from the action space (this is u_t)
        action = 0*env.action_space.sample()
        
        if flag == True:
            X[traj, t, :] = env.unwrapped._get_obs()
            U[traj, t, :] = action

        # Step the environment with the chosen action.
        # Note: the environment returns an observation in the default format,
        # so we use the internal state to record [theta, theta_dot].
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = env.unwrapped._get_obs()
        
        if flag == True:
            X_next[traj, t, :] = next_state
            # k = k+1

        # Update the current state.
        obs = next_state

        # (Optional) If the environment terminates early, you can break out.
        if terminated or truncated:
            break

    # Optional: print progress every 1000 trajectories.
    if (traj + 1) % 1000 == 0:
        print(f"Generated {traj + 1} / {n_traj} trajectories.")
    # print(X_next[traj, -1,:])
    # sys.exit()
# -------------------------
# Save the Data
# -------------------------

# After generating the data arrays X, U, and X_next:
# X shape: (n_traj, horizon, state_dim)
# U shape: (n_traj, horizon, action_dim)
# X_next shape: (n_traj, horizon, state_dim)

# Transpose the arrays so that state or action dimension is the first axis:
X_transposed = X.transpose(2, 0, 1)        # shape: (state_dim, n_traj, horizon)
U_transposed = U.transpose(2, 0, 1)        # shape: (action_dim, n_traj, horizon)
X_next_transposed = X_next.transpose(2, 0, 1)  # shape: (state_dim, n_traj, horizon)

# Now reshape the arrays so that the second and third dimensions collapse into one:
X_final = X_transposed.reshape(X_transposed.shape[0], -1)        # shape: (state_dim, n_traj * horizon)
U_final = U_transposed.reshape(U_transposed.shape[0], -1)        # shape: (action_dim, n_traj * horizon)
X_next_final = X_next_transposed.reshape(X_next_transposed.shape[0], -1)  # shape: (state_dim, n_traj * horizon)

# print(X_final[:,0])
# print(X_final[:,199])
# print(X_final[:,200])
# print(X_final[:,299])

# Save the arrays as a compressed NPZ file.
np.save('data/acrobot_seed/acrobot_seed'+str(seed_val)+'_x.npy', X_final)
np.save('data/acrobot_seed/acrobot_seed'+str(seed_val)+'_u.npy', U_final)
np.save('data/acrobot_seed/acrobot_seed'+str(seed_val)+'_y.npy', X_next_final)
# np.savez("pendulum_data_theta.npz", x=X, u=U, x_next=X_next)
# print("Data saved to pendulum_data_theta.npz")

env.close()
