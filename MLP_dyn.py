import os
import torch
from torch import nn
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, n_x, n_u):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_x+n_u, 32),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_x)
        )
    
    def forward(self, x, u):
        inp = torch.cat((x, u), dim=1).float()
        out = self.mlp(inp)
        return out


# -------------------------------
# Define a Dataset that forms (xₖ, uₖ, xₖ₊₁) samples
# -------------------------------
class StaticDataset(Dataset):
    def __init__(self, x, u, y):
        """
        x, u, y are tensors of shape:
          x: (n_obs, T)
          u: (n_act, T)
          y: (n_obs, T)  (where y represents xₖ₊₁)
          
        This Dataset creates samples so that for each time step k (from 0 to T-2):
          sample = (x[k], u[k], y[k+1])
        """
        # Transpose so that time is the first dimension:
        # x: (T, n_x), u: (T, n_u), y: (T, n_x)
        x = x.T  
        u = u.T  
        y = y.T  
        self.x_samples = x[:-1]
        self.u_samples = u[:-1]
        self.y_samples = y[1:]
    
    def __len__(self):
        return self.x_samples.shape[0]
    
    def __getitem__(self, idx):
        return self.x_samples[idx], self.u_samples[idx], self.y_samples[idx]


if __name__ == '__main__':

    # Set fixed random number seed
    torch.manual_seed(0)
    
    # torch.cuda.empty_cache()

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    print('Device : ', dev)
    device = torch.device(dev)

    # Prepare dataset
    model = 'cartpole'#'fish'
    train = True
    tspan = 400#200#600
    n_obs = 4#27
    n_act = 1#5
    seed = 0
    
   
    
    # FISH DATA
    x = torch.from_numpy(np.load('data/acrobot_seed/acrobot_seed'+str(seed)+'_x.npy')[:,:10000*400]).float().to(device) #[:,:10000*200])
    u = torch.from_numpy(np.load('data/acrobot_seed/acrobot_seed'+str(seed)+'_u.npy')[:,:10000*400]).float().to(device)
    y = torch.from_numpy(np.load('data/acrobot_seed/acrobot_seed'+str(seed)+'_y.npy')[:,:10000*400]).float().to(device)
    x[0, :] = x[0, :] - torch.pi * torch.ones(x[0, :].shape, device=x.device)
    y[0, :] = y[0, :] - torch.pi * torch.ones(y[0, :].shape, device=y.device)
    
    # Load testing data
    x_test = torch.from_numpy(np.load('data/acrobot_seed/acrobot_seed'+str(seed)+'_x.npy')[:,10000*400:12000*400]).float().to(device)
    u_test = torch.from_numpy(np.load('data/acrobot_seed/acrobot_seed'+str(seed)+'_u.npy')[:,10000*400:12000*400]).float().to(device)
    y_test = torch.from_numpy(np.load('data/acrobot_seed/acrobot_seed'+str(seed)+'_y.npy')[:,10000*400:12000*400]).float().to(device)
    x_test[0, :] = x_test[0, :] - torch.pi * torch.ones(x_test[0, :].shape, device=x_test.device)
    y_test[0, :] = y_test[0, :] - torch.pi * torch.ones(y_test[0, :].shape, device=y_test.device)
    
    print('Data Loaded')
    
    # ----- Create Dataset and DataLoader objects -----
    train_dataset = StaticDataset(x, u, y)
    test_dataset = StaticDataset(x_test, u_test, y_test)
    
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   

    
    inp = torch.cat((x,u))
    mlp_model = MLP(n_x=n_obs, n_u=n_act).to(device)
    # Initialize the MLP
    # mlp = torch.load('MLP_cptest_traj_'+per+'_4').to(device)
    # mlp = torch.load('MLP_fish_'+per+'pc').to(device)
    # mlp = torch.load('MLP_cartpole_5pc').to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4, weight_decay=1e-6)
    

    # Run the training loop
    
    
    if train:
        num_epochs = 100  # adjust as needed
        with open('epoch_loss_MLP_acrobot_'+str(seed)+'.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])
            
            for epoch in range(num_epochs):
                mlp_model.train()
                train_loss = 0.0
                for x_batch, u_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = mlp_model(x_batch, u_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * x_batch.size(0)
                train_loss /= len(train_dataset)
                
                mlp_model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for x_batch, u_batch, y_batch in test_loader:
                        outputs = mlp_model(x_batch, u_batch)
                        loss = criterion(outputs, y_batch)
                        test_loss += loss.item() * x_batch.size(0)
                test_loss /= len(test_dataset)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
                    
                # Log to CSV rows.
                # csv_rows.append([epoch, avg_loss, avg_test_loss, teacher_ratio])
                writer.writerow([epoch, train_loss, test_loss])
                f.flush()
        
        # Save the trained model
        torch.save(mlp_model, 'models/MLP/MLP_static_model_acrobot_'+str(seed)+'.pt')
        print("Training finished and model saved.")
        # sys.exit()
    
    # ----- Testing mode -----
    print('Generating trajectories to compare with GT')
    
    x0_1 = x_test[:,0]
    x0_2 = x_test[:,1200]
    u_t = u_test[:,400:800]
    
    # mlp0 = torch.load('models/MLP/MLP_static_model_pendulum_0.pt')
    mlp50 = torch.load('models/MLP/MLP_static_model_acrobot_0.pt')
    mlp100 = torch.load('models/MLP/MLP_static_model_acrobot_50.pt')
    mlp1000 = torch.load('models/MLP/MLP_static_model_acrobot_100.pt')
    
    # Initialize trajectory tensors with shape (tspan+1, n_obs)
    # y0    = torch.ones((tspan+1, n_obs)).to(device)
    y50   = torch.ones((tspan+1, n_obs)).to(device)
    y100  = torch.ones((tspan+1, n_obs)).to(device)
    y1000 = torch.ones((tspan+1, n_obs)).to(device)
    
    # Set the initial state (make sure x0_1 has shape (n_obs,))
    # y0[0, :]    = x0_2
    y50[0, :]   = x0_1
    y100[0, :]  = x0_1
    y1000[0, :] = x0_1
    
    # Generate trajectories using the MLP models
    for t in range(tspan):
        # Unsqueeze to add batch dimension: shape (1, n_obs) for state,
        # and (1, n_act) for control.
        # next_y0    = mlp0(y0[t].unsqueeze(0), u_t[:, t].unsqueeze(0))
        next_y50   = mlp50(y50[t].unsqueeze(0), u_t[:, t].unsqueeze(0))
        next_y100  = mlp100(y100[t].unsqueeze(0), u_t[:, t].unsqueeze(0))
        next_y1000 = mlp1000(y1000[t].unsqueeze(0), u_t[:, t].unsqueeze(0))
        # Remove the batch dimension and assign to the trajectory
        # y0[t+1]    = next_y0.squeeze(0)
        y50[t+1]   = next_y50.squeeze(0)
        y100[t+1]  = next_y100.squeeze(0)
        y1000[t+1] = next_y1000.squeeze(0)
    
    plt.figure()
    plt.plot(np.arange(tspan), (180/np.pi)*x_test[0, 0:400].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(tspan+1), y0[:, 0].detach().cpu().numpy(), 'g')
    plt.plot(np.arange(tspan+1), (180/np.pi)*y50[:, 0].detach().cpu().numpy(), 'b--')
    plt.plot(np.arange(tspan+1), (180/np.pi)*y100[:, 0].detach().cpu().numpy(), 'r--')
    plt.plot(np.arange(tspan+1), (180/np.pi)*y1000[:, 0].detach().cpu().numpy(), 'y--')
    plt.xlabel('Timesteps')
    plt.ylabel('Theta 1 (deg)')
    plt.legend(['Ground Truth','seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_theta1_traj.png')
    
    plt.figure()
    plt.plot(np.arange(tspan), (180/np.pi)*x_test[1, 0:400].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(tspan+1), y0[:, 0].detach().cpu().numpy(), 'g')
    plt.plot(np.arange(tspan+1), (180/np.pi)*y50[:, 1].detach().cpu().numpy(), 'b--')
    plt.plot(np.arange(tspan+1), (180/np.pi)*y100[:, 1].detach().cpu().numpy(), 'r--')
    plt.plot(np.arange(tspan+1), (180/np.pi)*y1000[:, 1].detach().cpu().numpy(), 'y--')
    plt.xlabel('Timesteps')
    plt.ylabel('Theta 2 (deg)')
    plt.legend(['Ground Truth','seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_theta2_traj.png')
    
    plt.figure()
    plt.plot(np.arange(tspan), x_test[2, 0:400].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(tspan+1), y0[:, 1].detach().cpu().numpy(), 'g')
    plt.plot(np.arange(tspan+1), y50[:, 2].detach().cpu().numpy(), 'b--')
    plt.plot(np.arange(tspan+1), y100[:, 2].detach().cpu().numpy(), 'r--')
    plt.plot(np.arange(tspan+1), y1000[:, 2].detach().cpu().numpy(), 'y--')
    plt.xlabel('Timesteps')
    plt.ylabel('Angular Velocity 1 (rad/s)')
    plt.legend(['Ground Truth','seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_omega1_traj.png')
    
    plt.figure()
    plt.plot(np.arange(tspan), x_test[3, 0:400].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(tspan+1), y0[:, 1].detach().cpu().numpy(), 'g')
    plt.plot(np.arange(tspan+1), y50[:, 3].detach().cpu().numpy(), 'b--')
    plt.plot(np.arange(tspan+1), y100[:, 3].detach().cpu().numpy(), 'r--')
    plt.plot(np.arange(tspan+1), y1000[:, 3].detach().cpu().numpy(), 'y--')
    plt.xlabel('Timesteps')
    plt.ylabel('Angular Velocity 2 (rad/s)')
    plt.legend(['Ground Truth','seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_omega2_traj.png')
    
    plt.figure()
    plt.plot(1,(180/np.pi)*x_test[0, 1].T.detach().cpu().numpy(), 'black', marker='o')
    plt.plot(1,(180/np.pi)*y50[1, 0].detach().cpu().numpy(), 'b', marker='o')
    plt.plot(1,(180/np.pi)*y100[1, 0].detach().cpu().numpy(), 'r', marker='o')
    plt.plot(1,(180/np.pi)*y1000[1, 0].detach().cpu().numpy(), 'y', marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Theta 1 (deg)')
    plt.legend(['Ground Truth', 'seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_theta1.png')
    
    plt.figure()
    plt.plot(1,(180/np.pi)*x_test[1, 1].T.detach().cpu().numpy(), 'black', marker='o')
    plt.plot(1,(180/np.pi)*y50[1, 1].detach().cpu().numpy(), 'b', marker='o')
    plt.plot(1,(180/np.pi)*y100[1, 1].detach().cpu().numpy(), 'r', marker='o')
    plt.plot(1,(180/np.pi)*y1000[1, 1].detach().cpu().numpy(), 'y', marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Theta 2 (deg)')
    plt.legend(['Ground Truth', 'seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_theta2.png')
    
    plt.figure()
    plt.plot(1,x_test[2, 1].T.detach().cpu().numpy(), 'black', marker='o')
    plt.plot(1,y50[1, 2].detach().cpu().numpy(), 'b', marker='o')
    plt.plot(1,y100[1, 2].detach().cpu().numpy(), 'r', marker='o')
    plt.plot(1,y1000[1, 2].detach().cpu().numpy(), 'y', marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Omega 1 (rad/s)')
    plt.legend(['Ground Truth', 'seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_omega1.png')
    plt.figure()
    plt.plot(1,x_test[3, 1].T.detach().cpu().numpy(), 'black', marker='o')
    plt.plot(1,y50[1, 3].detach().cpu().numpy(), 'b', marker='o')
    plt.plot(1,y100[1, 3].detach().cpu().numpy(), 'r', marker='o')
    plt.plot(1,y1000[1, 3].detach().cpu().numpy(), 'y', marker='o')
    plt.xlabel('Timesteps')
    plt.ylabel('Omega 2 (rad/s)')
    plt.legend(['Ground Truth', 'seed = 0','seed = 50','seed = 100'])
    plt.title('Comparing variance in pendulum: MLP')
    plt.savefig('acrobot_var_omega2.png')
    