import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import matplotlib.pyplot as plt
import csv

###############################################
# Model Definition
###############################################

class LSTMModel(nn.Module):
    def __init__(self, n_x, n_u, hidden_dim=64, num_layers=2):
        """
        n_x: state dimension
        n_u: control input dimension
        hidden_dim: number of hidden units in LSTM
        num_layers: number of LSTM layers
        """
        super(LSTMModel, self).__init__()
        self.input_dim = n_x + n_u  # Concatenate state and control each timestep.
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer (with batch_first=True)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        # Fully-connected layer to produce predicted next state.
        self.fc = nn.Linear(hidden_dim, n_x)
        
    def forward(self, x_seq, u_seq):
        """
        x_seq: (batch_size, seq_len, n_x)
        u_seq: (batch_size, seq_len, n_u)
        Returns: predicted state (batch_size, n_x) from the final timestep.
        """
        seq = torch.cat([x_seq, u_seq], dim=2)  # (batch_size, seq_len, n_x+n_u)
        batch_size = seq.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(seq.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(seq.device)
        lstm_out, _ = self.lstm(seq, (h0, c0))
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        output = self.fc(last_out)      # (batch_size, n_x)
        return output

###############################################
# Data Preparation Functions
###############################################

def create_sequences(x, u, y, seq_length):
    """
    Converts raw data (state, control, next state) into one-step sequences.
    Each sample uses a window of seq_length timesteps to predict the next state.
    
    x, u, y: tensors of shape (feature_dim, total_timesteps)
    Returns:
       x_seq: (num_samples, seq_length, n_x)
       u_seq: (num_samples, seq_length, n_u)
       y_target: (num_samples, n_x)  -- the state at time t+seq_length
    """
    x = x.T  # (total_timesteps, n_x)
    u = u.T  # (total_timesteps, n_u)
    y = y.T  # (total_timesteps, n_x)
    
    sequences_x = []
    sequences_u = []
    targets = []
    
    for i in range(len(x) - seq_length):
        sequences_x.append(x[i:i+seq_length])
        sequences_u.append(u[i:i+seq_length])
        targets.append(y[i+seq_length])
        
    sequences_x = torch.stack(sequences_x)
    sequences_u = torch.stack(sequences_u)
    targets = torch.stack(targets)
    
    return sequences_x, sequences_u, targets

def create_rollout_sequences(x, u, y, seq_length, rollout_steps):
    """
    Creates rollout-style sequences.
    For each sample, the first seq_length timesteps are the initial window,
    then the next rollout_steps timesteps are the ground truth rollout.
    
    x, u, y: tensors of shape (feature_dim, total_timesteps)
    Returns:
       x_seq: (num_samples, seq_length, n_x)
       u_seq: (num_samples, seq_length+rollout_steps, n_u)
       y_rollout: (num_samples, rollout_steps, n_x)
    """
    x = x.T  # (total_timesteps, n_x)
    u = u.T  # (total_timesteps, n_u)
    y = y.T  # (total_timesteps, n_x)
    
    sequences_x = []
    sequences_u = []
    sequences_y = []
    total_steps = x.shape[0]
    for i in range(total_steps - seq_length - rollout_steps):
        sequences_x.append(x[i:i+seq_length])
        sequences_u.append(u[i:i+seq_length+rollout_steps])
        sequences_y.append(y[i+seq_length:i+seq_length+rollout_steps])
        
    sequences_x = torch.stack(sequences_x)
    sequences_u = torch.stack(sequences_u)
    sequences_y = torch.stack(sequences_y)
    
    return sequences_x, sequences_u, sequences_y

class RolloutDataset(Dataset):
    def __init__(self, x_seq, u_seq, y_rollout):
        """
        x_seq: (num_samples, seq_length, n_x)
        u_seq: (num_samples, seq_length+rollout_steps, n_u)
        y_rollout: (num_samples, rollout_steps, n_x)
        """
        self.x_seq = x_seq
        self.u_seq = u_seq
        self.y_rollout = y_rollout
    def __len__(self):
        return self.x_seq.size(0)
    def __getitem__(self, idx):
        return self.x_seq[idx], self.u_seq[idx], self.y_rollout[idx]

###############################################
# Rollout Loss with Scheduled Sampling
###############################################

def rollout_loss(model, x_seq, u_full, y_rollout, teacher_forcing_ratio):
    """
    Performs a closed-loop rollout on a batch of samples and computes MSE loss.
    
    Parameters:
      x_seq: (B, seq_length, n_x) initial state window.
      u_full: (B, seq_length+rollout_steps, n_u) control inputs.
      y_rollout: (B, rollout_steps, n_x) ground truth rollout.
      teacher_forcing_ratio: probability in [0,1] to use ground truth state.
      
    Returns:
      loss: scalar loss value.
    """
    batch_size, seq_length, n_x = x_seq.shape
    rollout_steps = y_rollout.shape[1]
    predictions = []
    
    current_seq = x_seq.clone()  # (B, seq_length, n_x)
    
    for t in range(rollout_steps):
        u_window = u_full[:, t:t+seq_length, :]  # (B, seq_length, n_u)
        pred = model(current_seq, u_window)  # (B, n_x)
        predictions.append(pred)
        
        # Scheduled sampling: decide per sample whether to use ground truth.
        use_teacher = torch.rand(batch_size, device=pred.device) < teacher_forcing_ratio
        teacher_state = y_rollout[:, t, :]  # (B, n_x)
        next_state = torch.where(use_teacher.unsqueeze(1), teacher_state, pred)
        
        # Update current sequence: drop the first state and append next_state.
        current_seq = torch.cat([current_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)
    
    predictions = torch.stack(predictions, dim=1)  # (B, rollout_steps, n_x)
    loss = nn.MSELoss()(predictions, y_rollout)
    return loss

###############################################
# Trajectory Generation Function
###############################################

def generate_trajectory(model, x_init, u_traj, horizon, seq_length):
    """
    Generate a trajectory using the trained LSTM model.
    
    Parameters:
      x_init: (seq_length, n_x) initial window.
      u_traj: (horizon, n_u) control inputs.
      horizon: number of prediction steps.
      seq_length: window length the model expects.
      
    Returns:
      traj: (horizon, n_x) predicted states.
    """
    model.eval()
    traj = []
    current_seq = x_init.unsqueeze(0)  # (1, seq_length, n_x)
    
    for t in range(horizon):
        if t + seq_length <= u_traj.shape[0]:
            u_window = u_traj[t:t+seq_length].unsqueeze(0)
        else:
            last_u = u_traj[-1].unsqueeze(0)
            u_window = u_traj[t:].unsqueeze(0)
            missing = seq_length - u_window.shape[1]
            pad = last_u.repeat(1, missing, 1)
            u_window = torch.cat([u_window, pad], dim=1)
        with torch.no_grad():
            x_next = model(current_seq, u_window)
        traj.append(x_next.squeeze(0))
        current_seq = torch.cat([current_seq[:, 1:, :], x_next.unsqueeze(1)], dim=1)
    traj = torch.stack(traj)
    return traj

###############################################
# Main: Data Loading, Training, Testing, and CSV Logging
###############################################

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device : ', device)
    
    # --------------------------
    # Data Loading
    # --------------------------
    # Load training data (adjust file paths as needed)
    x = torch.from_numpy(np.load('data/undamped_pendulum_30ic_x.npy')[:,:4500*200]).float().to(device)
    u = torch.from_numpy(np.load('data/undamped_pendulum_30ic_u.npy')[:,:4500*200]).float().to(device)
    y = torch.from_numpy(np.load('data/undamped_pendulum_30ic_y.npy')[:,:4500*200]).float().to(device)
    x[0, :] = x[0, :] - torch.pi * torch.ones(x[0, :].shape, device=x.device)
    y[0, :] = y[0, :] - torch.pi * torch.ones(y[0, :].shape, device=y.device)
    
    # Load testing data
    x_test = torch.from_numpy(np.load('data/undamped_pendulum_30ic_x.npy')[:,4500*200:5000*200]).float().to(device)
    u_test = torch.from_numpy(np.load('data/undamped_pendulum_30ic_u.npy')[:,4500*200:5000*200]).float().to(device)
    y_test = torch.from_numpy(np.load('data/undamped_pendulum_30ic_y.npy')[:,4500*200:5000*200]).float().to(device)
    x_test[0, :] = x_test[0, :] - torch.pi * torch.ones(x_test[0, :].shape, device=x_test.device)
    y_test[0, :] = y_test[0, :] - torch.pi * torch.ones(y_test[0, :].shape, device=y_test.device)
    
    print('Data Loaded')
    
    # --------------------------
    # Create Rollout Sequences for Training & Testing
    # --------------------------
    seq_length = 20  # initial window length
    rollout_steps = 200 - seq_length  # desired rollout horizon
    
    x_seq, u_seq_full, y_rollout = create_rollout_sequences(x, u, y, seq_length, rollout_steps)
    x_test_seq, u_test_seq_full, y_test_rollout = create_rollout_sequences(x_test, u_test, y_test, seq_length, rollout_steps)
    
    print('Data Sequences Initialized')
    
    # Create Dataset objects.
    train_dataset = RolloutDataset(x_seq, u_seq_full, y_rollout)
    test_dataset = RolloutDataset(x_test_seq, u_test_seq_full, y_test_rollout)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --------------------------
    # Model Loading/Initialization
    # --------------------------
    lstm_model = torch.load('models/LSTM_pendulum_ic30_model_undamped.pt').to(device)
    # Alternatively, to initialize a new model:
    # lstm_model = LSTMModel(n_x=x.size(0), n_u=u.size(0), hidden_dim=64, num_layers=2).to(device)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=3e-4, weight_decay=1e-6)
    
    num_epochs = 50#50  # adjust as needed
    initial_teacher_ratio = 1.0
    final_teacher_ratio = 0.0
    
    ###############################################
    # Generate and Plot Trajectories
    ###############################################
    
    print('Generating trajectories to compare with Ground Truth')
    horizon = rollout_steps  # number of prediction steps
    
    # Trajectory 1: Use a test sequence at index 1200.
    x_init = x_test_seq[400]  # shape: (seq_length, n_x)
    u_traj = torch.zeros(horizon, u.size(0)).to(device)
    predicted_trajectory = generate_trajectory(lstm_model, x_init, u_traj, horizon, seq_length)
    print("Generated trajectory shape:", predicted_trajectory.shape)
    
    pred_traj_np = predicted_trajectory.cpu().numpy()
    pred_traj_full = np.zeros(np.shape(x_test[:,:200].T.cpu().numpy()))
    pred_traj_full[:seq_length,:] = x_test[:,400:400+seq_length].T.cpu().numpy()
    pred_traj_full[seq_length:,:] = pred_traj_np
    gt_traj = x_test[:, 400:400+seq_length+horizon].T.cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0] * (180 / np.pi), label='Ground Truth')
    plt.plot(pred_traj_full[:, 0] * (180 / np.pi), '--', label='LSTM Prediction')
    plt.xlabel('Time step')
    plt.ylabel('Pendulum angle (degree)')
    plt.title('LSTM: Ground Truth vs. Predicted Trajectory (Trajectory 1, $\\theta=\pm 30^{o}$)')
    plt.legend()
    plt.savefig('plots/LSTM_comp_roll_longer_5.png')
    
    # Trajectory 2: Use the first test sequence.
    x_init = x_test_seq[0]
    u_traj = torch.zeros(horizon, u.size(0)).to(device)
    predicted_trajectory = generate_trajectory(lstm_model, x_init, u_traj, horizon, seq_length)
    print("Generated trajectory shape:", predicted_trajectory.shape)
    
    pred_traj_np = predicted_trajectory.cpu().numpy()
    pred_traj_full = np.zeros(np.shape(x_test[:,:200].T.cpu().numpy()))
    pred_traj_full[:seq_length,:] = x_test[:,:seq_length].T.cpu().numpy()
    pred_traj_full[seq_length:,:] = pred_traj_np
    gt_traj = x_test[:, :seq_length+horizon].T.cpu().numpy()
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0] * (180 / np.pi), label='Ground Truth')
    plt.plot(pred_traj_full[:, 0] * (180 / np.pi), '--', label='LSTM Prediction')
    plt.xlabel('Time step')
    plt.ylabel('Pendulum angle (degree)')
    plt.title('LSTM: Ground Truth vs. Predicted Trajectory (Trajectory 2, $\\theta=\pm 30^{o}$)')
    plt.legend()
    plt.savefig('plots/LSTM_comp_roll_longer_6.png')
    
    sys.exit()
