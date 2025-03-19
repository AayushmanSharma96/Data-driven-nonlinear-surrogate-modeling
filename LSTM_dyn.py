import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, n_x, n_u, hidden_dim=64, num_layers=2):
        """
        n_x: state dimension
        n_u: control input dimension
        hidden_dim: number of hidden units in LSTM
        num_layers: number of LSTM layers
        """
        super(LSTMModel, self).__init__()
        self.input_dim = n_x #+ n_u  # we will concatenate state and control at each time step
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The LSTM layer (note: batch_first=True means input shape is (batch, seq_len, feature_dim))
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        # A final fully-connected layer to produce the predicted next state
        self.fc = nn.Linear(hidden_dim, n_x)
        
    def forward(self, x_seq, u_seq):
        """
        x_seq: tensor of shape (batch_size, seq_len, n_x)
        u_seq: tensor of shape (batch_size, seq_len, n_u)
        Returns: predicted state (usually from the last time step) of shape (batch_size, n_x)
        """
        # Concatenate the state and control inputs along the feature dimension:
        # (batch_size, seq_len, n_x+n_u)
        seq = x_seq#torch.cat([x_seq, u_seq], dim=2)
        
        # Initialize hidden and cell states to zero
        batch_size = seq.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(seq.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(seq.device)
        
        # Pass the sequence through the LSTM
        lstm_out, _ = self.lstm(seq, (h0, c0))
        
        # Here we choose to use the LSTM output at the final time step.
        last_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)
        
        # Map to the output state dimension
        output = self.fc(last_out)  # shape: (batch_size, n_x)
        return output

def create_sequences(x, u, y, seq_length):
    """
    Converts raw data (state, control, next state) into sequences.
    Each training sample will use a window of seq_length time steps 
    to predict the next state.
    
    x, u, y: tensors of shape (feature_dim, total_timesteps)
    Returns:
       x_seq: (num_samples, seq_length, n_x)
       u_seq: (num_samples, seq_length, n_u)
       y_target: (num_samples, n_x)
    """
    # Transpose so that timesteps are the first dimension
    x = x.T  # shape: (total_timesteps, n_x)
    u = u.T  # shape: (total_timesteps, n_u)
    y = y.T  # shape: (total_timesteps, n_x)
    
    sequences_x = []
    sequences_u = []
    targets = []
    
    # We go from time index 0 up to total_timesteps - seq_length.
    for i in range(len(x) - seq_length):
        sequences_x.append(x[i:i+seq_length])
        sequences_u.append(u[i:i+seq_length])
        # The target is the state right after the end of the sequence.
        targets.append(y[i+seq_length])
        
    # Stack the lists into tensors.
    sequences_x = torch.stack(sequences_x)  # (num_samples, seq_length, n_x)
    sequences_u = torch.stack(sequences_u)  # (num_samples, seq_length, n_u)
    targets = torch.stack(targets)          # (num_samples, n_x)
    
    return sequences_x, sequences_u, targets

# Define a custom Dataset for our sequence data
class SequenceDataset(Dataset):
    def __init__(self, x_seq, u_seq, y_target):
        """
        x_seq: Tensor of shape (num_samples, seq_length, n_x)
        u_seq: Tensor of shape (num_samples, seq_length, n_u)
        y_target: Tensor of shape (num_samples, n_x)
        """
        self.x_seq = x_seq
        self.u_seq = u_seq
        self.y_target = y_target

    def __len__(self):
        return self.x_seq.size(0)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.u_seq[idx], self.y_target[idx]

def generate_trajectory(model, x_init, u_traj, horizon, seq_length):
    """
    Generate a trajectory using the trained LSTM model.
    
    Parameters:
      model:         The trained LSTM model.
      x_init:        A tensor of shape (seq_length, n_x) containing the initial states.
      u_traj:        A tensor of shape (horizon, n_u) containing control inputs 
                     for each future step.
      horizon:       Number of prediction steps.
      seq_length:    The sequence length that the model expects.
      
    Returns:
      traj:          A tensor of shape (horizon, n_x) containing the predicted trajectory.
    """
    model.eval()  # set model to evaluation mode
    traj = []     # list to store predicted states
    
    # Ensure x_init has a batch dimension: shape (1, seq_length, n_x)
    current_seq = x_init.unsqueeze(0)
    
    # Generate a trajectory of 'horizon' steps.
    for t in range(horizon):
        # Create a control window for the current time step.
        if t + seq_length <= u_traj.shape[0]:
            u_window = u_traj[t : t+seq_length].unsqueeze(0)  # shape: (1, seq_length, n_u)
        else:
            # If not enough control inputs remain, pad with the last control input.
            last_u = u_traj[-1].unsqueeze(0)  # shape: (1, n_u)
            u_window = u_traj[t:].unsqueeze(0)
            missing = seq_length - u_window.shape[1]
            pad = last_u.repeat(1, missing, 1)
            u_window = torch.cat([u_window, pad], dim=1)
        
        with torch.no_grad():
            x_next = model(current_seq, u_window)  # shape: (1, n_x)
        
        traj.append(x_next.squeeze(0))
        
        # Update current sequence by dropping the first state and appending the new prediction.
        current_seq = torch.cat([current_seq[:, 1:, :], x_next.unsqueeze(1)], dim=1)
    
    traj = torch.stack(traj)
    return traj

if __name__ == '__main__':
    # Set fixed random seed
    torch.manual_seed(0)
    
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    print('Device : ', dev)
    device = torch.device(dev)

    # Prepare dataset
    # For example, load your data from numpy files (adjust file names and paths as needed)
    x = torch.from_numpy(np.load('data/undamped_pendulum_10ic_large_x.npy')[:,:4500*200]).float().to(device)
    u = torch.from_numpy(np.load('data/undamped_pendulum_10ic_large_u.npy')[:,:4500*200]).float().to(device)
    y = torch.from_numpy(np.load('data/undamped_pendulum_10ic_large_y.npy')[:,:4500*200]).float().to(device)
    
    x[0,:] = x[0,:]-torch.pi * torch.ones(x[0, :].shape, device=x.device)
    y[0,:] = y[0,:]-torch.pi * torch.ones(y[0, :].shape, device=y.device)
    # Load Testing Data
    x_test = torch.from_numpy(np.load('data/undamped_pendulum_10ic_large_x.npy')[:,4500*200:5000*200]).float().to(device)
    u_test = torch.from_numpy(np.load('data/undamped_pendulum_10ic_large_u.npy')[:,4500*200:5000*200]).float().to(device)
    y_test = torch.from_numpy(np.load('data/undamped_pendulum_10ic_large_y.npy')[:,4500*200:5000*200]).float().to(device)
    
    x_test[0,:] = x_test[0,:]-torch.pi * torch.ones(x_test[0, :].shape, device=x_test.device)
    y_test[0,:] = y_test[0,:]-torch.pi * torch.ones(y_test[0, :].shape, device=y_test.device)
    
    print('Data Loaded')
    
    seq_length = 50  # set your desired sequence length

    # Create sequences from your training data (x, u, y)
    x_seq, u_seq, y_target = create_sequences(x, u, y, seq_length)

    # Similarly, create sequences for your test data.
    x_test_seq, u_test_seq, y_test_target = create_sequences(x_test, u_test, y_test, seq_length)
    
    print('Data Sequences Initialized')
    
    # Create Dataset objects for training and testing
    train_dataset = SequenceDataset(x_seq, u_seq, y_target)
    test_dataset = SequenceDataset(x_test_seq, u_test_seq, y_test_target)
    
    # Create DataLoader objects
    batch_size = 256  # adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
   
    
    # Uncomment if you want to initialize the model
    lstm_model = LSTMModel(n_x=x.size(0), n_u=u.size(0), hidden_dim=32, num_layers=2).to(device)
    # Alternatively, load your pre-trained model:
    # lstm_model = torch.load('models/LSTM_pendulum_ic_model_undamped.pt').to(device)
   
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        lstm_model.parameters(), lr=1e-3, weight_decay=1e-5
    )
    
    
    num_epochs = 20  # adjust as needed
    for epoch in range(num_epochs):
        lstm_model.train()
        epoch_loss = 0.0
        
        if epoch==250:
            optimizer = torch.optim.Adam(
                lstm_model.parameters(), lr=1e-4, weight_decay=1e-6
            )
            
        if epoch==400:
            optimizer = torch.optim.Adam(
                lstm_model.parameters(), lr=3e-5, weight_decay=1e-7
            )
        # Iterate over minibatches using the DataLoader
        for batch in train_loader:
            x_batch, u_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = lstm_model(x_batch, u_batch)  # outputs shape: (batch_size, n_x)
            loss = loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        avg_loss = epoch_loss / len(train_dataset)
        
        # Optionally, evaluate on test data
        lstm_model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x_batch, u_batch, y_batch = batch
                test_out = lstm_model(x_batch, u_batch)
                test_loss += loss_function(test_out, y_batch).item() * x_batch.size(0)
        avg_test_loss = test_loss / len(test_dataset)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
    
    # Save the model if desired
    torch.save(lstm_model, 'models/LSTM_pendulum_ic_model_undamped_test10.pt')
    print('Training process has finished.')
    
    # for t in range(20):
    #     print('init = ', x_test[0, 200*t])
    
    print('Generating trajectories to compare with GT')
    # Suppose you have a trained model and you want to generate a trajectory.
    horizon = 200 - seq_length  # number of prediction steps
    
    # Use the first test sequence as the initial window.
    x_init = x_test_seq[0+1200]  # shape: (seq_length, n_x)
    
    # Create a control trajectory (e.g., zeros)
    u_traj = torch.zeros(horizon, u.size(0)).to(device)  # shape: (horizon, n_u)
    
    predicted_trajectory = generate_trajectory(lstm_model, x_init, u_traj, horizon, seq_length)
    print("Generated trajectory shape:", predicted_trajectory.shape)
    
    # Plotting
    pred_traj_np = predicted_trajectory.cpu().numpy()
    gt_traj = x_test[:, 1200+0:200+1200].T.cpu().numpy()  # adjust slicing as needed
    pred = np.zeros(np.shape(gt_traj))
    pred[:seq_length, :] = gt_traj[:seq_length, :]
    pred[seq_length:, :] = pred_traj_np
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0] * (180 / np.pi), label='Ground Truth')
    plt.plot(pred[:, 0] * (180 / np.pi), '--', label='LSTM Prediction')
    plt.xlabel('Time step')
    plt.ylabel('Pendulum angle (degree)')
    plt.title('LSTM: Ground Truth vs. Predicted Trajectory for Pendulum')
    plt.legend()
    plt.savefig('plots/LSTM_comp10_zero_3.png')
    
    
    x_init = x_test_seq[0]  # shape: (seq_length, n_x)
    
    # Create a control trajectory (e.g., zeros)
    u_traj = torch.zeros(horizon, u.size(0)).to(device)  # shape: (horizon, n_u)
    
    predicted_trajectory = generate_trajectory(lstm_model, x_init, u_traj, horizon, seq_length)
    print("Generated trajectory shape:", predicted_trajectory.shape)
    
    # Plotting
    pred_traj_np = predicted_trajectory.cpu().numpy()
    gt_traj = x_test[:, 0:200].T.cpu().numpy()  # adjust slicing as needed
    pred = np.zeros(np.shape(gt_traj))
    pred[:seq_length, :] = gt_traj[:seq_length, :]
    pred[seq_length:, :] = pred_traj_np
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0] * (180 / np.pi), label='Ground Truth')
    plt.plot(pred[:, 0] * (180 / np.pi), '--', label='LSTM Prediction')
    plt.xlabel('Time step')
    plt.ylabel('Pendulum angle (degree)')
    plt.title('LSTM: Ground Truth vs. Predicted Trajectory for Pendulum')
    plt.legend()
    plt.savefig('plots/LSTM_comp10_zero_4.png')
    
    sys.exit()
