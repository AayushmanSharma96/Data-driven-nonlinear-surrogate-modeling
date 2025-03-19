import os
import torch
from torch import nn

# from torch.utils.data import DataLoader
# from torchvision import transforms
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
# import seaborn as sns


class MLP(nn.Module):
    def __init__(self, n_x, n_u):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_x+n_u, 32),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(16, 32),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Dropout(p=0.6),
            nn.Linear(64, n_x)
        )
    
    def forward(self, x, u):
        inp = torch.cat((x, u), dim=1).float()
        out = self.mlp(inp)
        return out

'''class MLP(nn.Module):
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
        inp = torch.cat((x,u)).float()
        residual = x.T
        out = self.mlp(inp.T)
        out+= residual
        return out
'''    

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
    # mlp = MLP(torch.Tensor.size(x)[0], torch.Tensor.size(u)[0]).to(device)
    # mlp = torch.load('MLP_cptest_traj_'+per+'_4').to(device)
    # mlp = torch.load('MLP_fish_'+per+'pc').to(device)
    # mlp = torch.load('MLP_cartpole_5pc').to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    
    # def loss_function(outputs, y):
    #     return torch.linalg.norm(outputs-y,2)**2
    
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
    '''else:
        mlp_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, u_batch, y_batch in test_loader:
                outputs = mlp_model(x_batch, u_batch)
                total_loss += loss_function(outputs, y_batch) * x_batch.size(0)
        total_loss /= len(test_dataset)
        print("Test Loss:", total_loss.item())
            

        # Process is complete.
        print('Training process has finished.')
        
        
        print('loss = ',loss)
        print(mlp(x,u).size())
        li = []
        for params in mlp.parameters():
            li.append(params)
        
        for t in range(2):
            a = torch.autograd.grad(mlp(x,u)[t], mlp.parameters(), grad_outputs=torch.ones_like(mlp(x,u)[t]))
            b = torch.zeros((0)).to(device)
            for stuff in a:
                stuff = stuff.flatten()
                b= torch.cat([b,stuff],-1)
            if t==0:
                lis = b.view(1,4516)
            else:
                lis = torch.cat([lis,b.view(1,4516)],0)
        print(lis.size())
        sys.exit()
        torch.save(mlp, 'MLP_fish_'+per)
        # torch.save(mlp, 'MLP_cptest_RESNET_'+per+'pc2')
        # torch.save(mlp, 'MLP_cptest_variance_'+per+'_'+seed+'_3')
        
        
        sys.exit()'''
        
        

    
    # Testing loop
    # testx = .5*torch.ones(1).to(device)
    # testu = .3*torch.ones(1).to(device)
    # print(torch.sin(testx)+testu)
    # mlp = torch.load('MLP_sine')
    # print(mlp(testx,testu))
    # sys.exit()
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
    
    # plt.figure()
    # plt.plot(np.arange(2), x_test[1, 600:602].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(2), y50[:2, 1].detach().cpu().numpy(), 'b')
    # plt.plot(np.arange(2), y100[:2, 1].detach().cpu().numpy(), 'r')
    # plt.plot(np.arange(2), y1000[:2, 1].detach().cpu().numpy(), 'y')
    # plt.xlabel('Timesteps')
    # plt.ylabel('Angular Velocity (rad/s)')
    # plt.legend(['seed = 0','seed = 50','seed = 100'])
    # plt.title('Comparing variance in pendulum: MLP')
    # plt.savefig('pendulum5_var_omega2.png')
    
    # plt.figure()
    # plt.plot(np.arange(tspan), x_test[2, 1200:1400].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(tspan+1), y0[:, 2].detach().cpu().numpy(), 'g')
    # plt.plot(np.arange(tspan+1), y50[:, 2].detach().cpu().numpy(), 'b')
    # plt.plot(np.arange(tspan+1), y100[:, 2].detach().cpu().numpy(), 'r')
    # plt.plot(np.arange(tspan+1), y1000[:, 2].detach().cpu().numpy(), 'y')
    # plt.xlabel('Timesteps')
    # plt.ylabel('Cart velocity')
    # plt.legend(['Ground Truth','seed = 0','seed = 50','seed = 100','seed = 1000'])
    # plt.title('Comparing variance in cartpole: MLP')
    # plt.savefig('cartpole_var_xdot_traj.png')
    
    # plt.figure()
    # plt.plot(np.arange(tspan), x_test[3, 1200:1400].T.detach().cpu().numpy(), 'black')
    # plt.plot(np.arange(tspan+1), y0[:, 3].detach().cpu().numpy(), 'g')
    # plt.plot(np.arange(tspan+1), y50[:, 3].detach().cpu().numpy(), 'b')
    # plt.plot(np.arange(tspan+1), y100[:, 3].detach().cpu().numpy(), 'r')
    # plt.plot(np.arange(tspan+1), y1000[:, 3].detach().cpu().numpy(), 'y')
    # plt.xlabel('Timesteps')
    # plt.ylabel('Cart position')
    # plt.legend(['Ground Truth','seed = 0','seed = 50','seed = 100','seed = 1000'])
    # plt.title('Comparing variance in cartpole: MLP')
    # plt.savefig('cartpole_var_thetadot_traj.png')
    
    sys.exit()    
        
    
    
    seed = '0'
    x_0 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_x.npy')[:,:900*30]).float().to(device)
    u_0 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_u.npy')[:,:900*30]).float().to(device)
    y_0 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_xkp1.npy')[:,:900*30]).float().to(device)
    seed = '50'
    x_50 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_x.npy')[:,:900*30]).float().to(device)
    u_50 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_u.npy')[:,:900*30]).float().to(device)
    y_50 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_xkp1.npy')[:,:900*30]).float().to(device)
    seed = '69'
    x_69 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_x.npy')[:,:900*30]).float().to(device)
    u_69 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_u.npy')[:,:900*30]).float().to(device)
    y_69 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_xkp1.npy')[:,:900*30]).float().to(device)
    seed = '100'
    x_100 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_x.npy')[:,:900*30]).float().to(device)
    u_100 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_u.npy')[:,:900*30]).float().to(device)
    y_100 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_xkp1.npy')[:,:900*30]).float().to(device)
    seed = '500'
    x_500 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_x.npy')[:,:900*30]).float().to(device)
    u_500 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_u.npy')[:,:900*30]).float().to(device)
    y_500 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_xkp1.npy')[:,:900*30]).float().to(device)
    seed = '1000'
    x_1000 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_x.npy')[:,:900*30]).float().to(device)
    u_1000 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_u.npy')[:,:900*30]).float().to(device)
    y_1000 = torch.from_numpy(np.load('data/'+seed+'TESTeps'+per+'pc_'+model+'NN_xkp1.npy')[:,:900*30]).float().to(device)
    
    
    seedlist = ['50','50','69','100','500','1000']
    for seed in seedlist:
        mlp_array = ['MLP_cptest_variance_'+per+'_'+seed+'_00','MLP_cptest_variance_'+per+'_'+seed+'_01','MLP_cptest_variance_'+per+'_'+seed+'_02']
        k = 0
        print('Current seed = '+seed)
        for item in mlp_array:
            mlp = torch.load(item).to(device)
            k+=1
            H = np.load('H_Cartpole_'+per+'_'+seed+'_'+str(k)+'.npy')
            
            h0 = mlp(x_0,u_0)
            delta0 = (y_0.T - h0).flatten().detach().cpu().numpy()
            F_mat0 = H.T @ delta0
            print('delta0 = ', np.max(delta0))
            print('normcheck = ', np.linalg.norm(H,2))
            print(np.max(H))
            print('Fmat = ', np.max(H.T @ (y_0.T).flatten().detach().cpu().numpy()))
            sys.exit()
            
            h50 = mlp(x_50,u_50)
            delta50 = (y_50.T - h50).flatten().detach().cpu().numpy()
            F_mat50 = H.T @ delta50
            
            h69 = mlp(x_69,u_69)
            delta69 = (y_69.T - h69).flatten().detach().cpu().numpy()
            F_mat69 = H.T @ delta69
            
            h100 = mlp(x_100,u_100)
            delta100 = (y_100.T - h100).flatten().detach().cpu().numpy()
            F_mat100 = H.T @ delta100
            
            h500 = mlp(x_500,u_500)
            delta500 = (y_500.T - h500).flatten().detach().cpu().numpy()
            F_mat500 = H.T @ delta500
            
            h1000 = mlp(x_1000,u_1000)
            delta1000 = (y_1000.T - h1000).flatten().detach().cpu().numpy()
            F_mat1000 = H.T @ delta1000
            
            np.save('1H_tr_delta_'+str(k)+'_'+per+'pc'+seed+'_0.npy', F_mat0)
            np.save('1H_tr_delta_'+str(k)+'_'+per+'pc'+seed+'_50.npy', F_mat50)
            np.save('1H_tr_delta_'+str(k)+'_'+per+'pc'+seed+'_69.npy', F_mat69)
            np.save('1H_tr_delta_'+str(k)+'_'+per+'pc'+seed+'_100.npy', F_mat100)
            np.save('1H_tr_delta_'+str(k)+'_'+per+'pc'+seed+'_500.npy', F_mat500)
            np.save('1H_tr_delta_'+str(k)+'_'+per+'pc'+seed+'_1000.npy', F_mat1000)
            
            '''for t in range(108000):
                a = torch.autograd.grad(mlp(x,u).flatten()[t], mlp.parameters(), grad_outputs=torch.ones_like(mlp(x,u).flatten()[t]))
                b = torch.zeros((0)).to(device)
                for stuff in a:
                    stuff = stuff.flatten()
                    b= torch.cat([b,stuff],-1)
                if t==0:
                    lis = b.view(1,2564)
                else:
                    lis = torch.cat([lis,b.view(1,2564)],0)
        
            np.save('H_Cartpole_'+per+'_'+seed+'_'+str(k)+'.npy',lis.detach().cpu().numpy())'''
    
    sys.exit()
    # mlp = torch.load('MLP_cptest_RESNET_'+per+'pc2').to(device)
    # output = mlp(x,u).flatten()
    
    '''for t in range(108000):
        a = torch.autograd.grad(mlp(x,u).flatten()[t], mlp.parameters(), grad_outputs=torch.ones_like(mlp(x,u).flatten()[t]))
        b = torch.zeros((0)).to(device)
        for stuff in a:
            stuff = stuff.flatten()
            b= torch.cat([b,stuff],-1)
        if t==0:
            lis = b.view(1,2564)
        else:
            lis = torch.cat([lis,b.view(1,2564)],0)
    print(lis.size())
    # np.save('Cartpole_TestLoss_'+per+'pc.npy',np.array(test_loss_list))
    # np.save('Cartpole_TrainLoss_'+per+'pc.npy',np.array(train_loss_list))
    np.save('H_Phi_Cartpole_'+per+'pc_resnet_fixed.npy',lis.detach().cpu().numpy())
    H_phi = lis.detach().cpu().numpy()'''
    
   
    H_phi = np.load('H_Phi_Cartpole_'+per+'pc_resnet_fixed.npy')#np.load('H_Phi_Cartpole_25pc_fixed.npy')
    
    '''trHmat = (H_phi.T) @ H_phi
    np.save('H_tr_H__'+per+'pc_resnet_fixed.npy', trHmat)
    sys.exit()'''
    
    h = mlp(x,u)#.detach().cpu().numpy()
    delta = (y.T - h).flatten().detach().cpu().numpy()
    F_mat = H_phi.T @ delta
    #F = (y.T).detach().cpu().numpy()
    np.save('H_tr_delta_'+per+'pc'+seed+'.npy', F_mat)
    print(np.shape(F_mat))
    
    # np.save('H_tr_f_1pc.npy', H_phi.T @ F)
    # np.save('H_tr_h_1pc.npy', H_phi.T @ H)
    
    sys.exit()
        
    # x_test1 = torch.from_numpy(np.load(seed+'_eps5pc_cartpoleNN_x.npy')[:,900*30:900*31]).float().to(device)
    # u_test1 = torch.from_numpy(np.load(seed+'_eps5pc_cartpoleNN_u.npy')[:,900*30:900*31]).float().to(device)
    # y_test1 = torch.from_numpy(np.load(seed+'_eps5pc_cartpoleNN_xkp1.npy')[:,900*30:900*31]).float().to(device)
    
    
    '''seed_list = ['20','40', '60', '80', '100', '150']
    np.save('data/ygt.npy',y_test1.detach().cpu().numpy())
    np.save('data/x00.npy',x_test1.detach().cpu().numpy())
    for seed_curr in seed_list:
        mlp = torch.load('MLP_cartpole_5pc_'+seed_curr).to(device)
        y_pred = mlp(x_test1,u_test1)
        print(y_pred)
        np.save('data/'+seed_curr+'_ypred.npy',y_pred.detach().cpu().numpy())'''
    
    
    # y_pred_local = mlp(x_test,u_test).detach().cpu().numpy()
    # y_pred_larger = mlp(x_test2,u_test2).detach().cpu().numpy()
    # np.save('y_pred_larger.npy',y_pred_larger)
    for t in range(1):
        per = '10'#str(int((t+1)))
        
        x_test = torch.from_numpy(np.load('data/120eps'+per+'pc_'+model+'NN_x.npy')[:,900*30:]).float().to(device)
        u_test = torch.from_numpy(np.load('data/120eps'+per+'pc_'+model+'NN_u.npy')[:,900*30:]).float().to(device)
        y_test = torch.from_numpy(np.load('data/120eps'+per+'pc_'+model+'NN_xkp1.npy')[:,900*30:]).float().to(device)
        
        # FISH TESTING
        # x_test = torch.from_numpy(np.load('data/eps'+per+'pc_'+model+'NN_x.npy')[:,600*450:]).float().to(device)
        # u_test = torch.from_numpy(np.load('data/eps'+per+'pc_'+model+'NN_u.npy')[:,600*450:]).float().to(device)
        # y_test = torch.from_numpy(np.load('data/eps'+per+'pc_'+model+'NN_xkp1.npy')[:,600*450:]).float().to(device)
        
        x_0 = x_test
        u = u_test
        y_gt = y_test
        
        x_kp1 = torch.ones((tspan+1,n_obs)).to(device)#mlp(x_0[:,20], u[:,20])
        x_kp2 = torch.ones((tspan+1,n_obs)).to(device)#mlp(x_0[:,20], u[:,20])
        x_k = torch.ones((tspan+1,n_obs)).to(device)
        # x_kp1 = torch.ones((201,2)).to(device)
        x_k2 = torch.ones((tspan+1,n_obs)).to(device)
        # x_kp12 = torch.ones((201,2)).to(device)
        # mlp_nl = torch.load('MLP_cart_'+per+'pc')
        mlp_nl = torch.load('MLP_fish_'+per+'pc2')
        # mlp_lin = torch.load('MLP_pendulum_ltest')
        # x_kp1 = mlp(x_test, u_test)
        
        # np.save('traj_xkp1_'+deg+'_pen.npy',x_kp1.detach().cpu().numpy())
        # np.save('traj_xk_'+deg+'_pen.npy',x_test.detach().cpu().numpy())
        # np.save('traj_y_'+deg+'_pen.npy',y_test.detach().cpu().numpy())
        
        x_0 = x_test[:,tspan*5]
        u = u_test[:, tspan*5:tspan*6]
        x_k[0] = x_0
        # x_k2[0] = x_0
        
        for t in range(tspan):
            x_kp1[t] = mlp_nl(x_k[t],u[:,t]).flatten()
            x_k[t+1] = x_kp1[t]
            
            # x_kp2[t] = mlp_lin(x_k2[t],u[:,t]).flatten()
            # x_k2[t+1] = x_kp2[t]
        
        
        # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
        # np.save('traj_xk_'+deg+'_cartgt1.npy',x_test[:,:100].detach().cpu().numpy())
        # np.save('traj_xk_'+deg+'_carttraj1.npy',x_k.detach().cpu().numpy())
        
        np.save('2traj_xk_'+model+per+'pcgt1.npy',x_test[:,tspan*5:tspan*6].detach().cpu().numpy())
        np.save('2traj_xk_'+model+per+'pctraj1.npy',x_k.detach().cpu().numpy())
        # np.save('traj_xk_penl4.npy',x_k2.detach().cpu().numpy())
        
        
        x_0 = x_test[:,tspan*8]
        u = u_test[:, tspan*8:tspan*9]
        x_k[0] = x_0
        # x_k2[0] = x_0
        
        for t in range(tspan):
            x_kp1[t] = mlp_nl(x_k[t],u[:,t]).flatten()
            x_k[t+1] = x_kp1[t]
            
            # x_kp2[t] = mlp_lin(x_k2[t],u[:,t]).flatten()
            # x_k2[t+1] = x_kp2[t]
        
        
        # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
        # np.save('traj_xk_'+deg+'_cartgt1.npy',x_test[:,:100].detach().cpu().numpy())
        # np.save('traj_xk_'+deg+'_carttraj1.npy',x_k.detach().cpu().numpy())
        
        np.save('2traj_xk_'+model+per+'pcgt2.npy',x_test[:,tspan*8:tspan*9].detach().cpu().numpy())
        np.save('2traj_xk_'+model+per+'pctraj2.npy',x_k.detach().cpu().numpy())
        
    
    sys.exit()
    x_0 = x_test[:,400]
    u = u_test
    x_k[0] = x_0
    x_k2[0] = x_0
    
    for t in range(30):
        x_kp1[t] = mlp_nl(x_k[t],u[:,t]).flatten()
        x_k[t+1] = x_kp1[t]
        
        x_kp2[t] = mlp_lin(x_k2[t],u[:,t]).flatten()
        x_k2[t+1] = x_kp2[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    # np.save('traj_xk_'+deg+'_cartgt1.npy',x_test[:,:100].detach().cpu().numpy())
    # np.save('traj_xk_'+deg+'_carttraj1.npy',x_k.detach().cpu().numpy())
    
    np.save('traj_xk_pengt2.npy',x_test[:,400:600].detach().cpu().numpy())
    np.save('traj_xk_pennl2.npy',x_k.detach().cpu().numpy())
    np.save('traj_xk_penl2.npy',x_k2.detach().cpu().numpy())
    
    
    x_0 = x_test[:,1400]
    u = u_test
    x_k[0] = x_0
    x_k2[0] = x_0
    
    for t in range(200):
        x_kp1[t] = mlp_nl(x_k[t],u[:,t]).flatten()
        x_k[t+1] = x_kp1[t]
        
        x_kp2[t] = mlp_lin(x_k2[t],u[:,t]).flatten()
        x_k2[t+1] = x_kp2[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    # np.save('traj_xk_'+deg+'_cartgt1.npy',x_test[:,:100].detach().cpu().numpy())
    # np.save('traj_xk_'+deg+'_carttraj1.npy',x_k.detach().cpu().numpy())
    
    np.save('traj_xk_pengt3.npy',x_test[:,1400:1600].detach().cpu().numpy())
    np.save('traj_xk_pennl3.npy',x_k.detach().cpu().numpy())
    np.save('traj_xk_penl3.npy',x_k2.detach().cpu().numpy())
    sys.exit()    
    
    x_0 = x_test[:,400]
    u = u_test
    x_k[0] = x_0
    x_k2[0] = x_0
    for t in range(100):
        x_kp1[t] = mlp(x_k[t],u[:,t]).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_cartgt2.npy',x_test[:,400:500].detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_carttraj2.npy',x_k.detach().cpu().numpy())
    
    
    x_0 = x_test[:,900]
    u = u_test
    x_k[0] = x_0
    for t in range(100):
        x_kp1[t] = mlp(x_k[t],u[:,t]).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_cartgt3.npy',x_test[:,900:1000].detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_carttraj3.npy',x_k.detach().cpu().numpy())
    sys.exit()
    
    x_0 = x_test[:,0]
    x_0[0] = 0
    x_0[1] = 20*np.pi/180
    u = 0*u_test[:,0]
    x_k[0] = x_0
    for t in range(200):
        x_kp1[t] = mlp(x_k[t],u).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_carttest20.npy',x_k.detach().cpu().numpy())
   
    x_0 = x_test[:,0]
    x_0[0] = 0
    x_0[1] = 45*np.pi/180
    u = 0*u_test[:,0]
    x_k[0] = x_0
    for t in range(200):
        x_kp1[t] = mlp(x_k[t],u).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_carttest45.npy',x_k.detach().cpu().numpy())
    
    x_0 = x_test[:,0]
    x_0[0] = 0
    x_0[1] = 90*np.pi/180
    u = 0*u_test[:,0]
    x_k[0] = x_0
    for t in range(200):
        x_kp1[t] = mlp(x_k[t],u).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_carttest90.npy',x_k.detach().cpu().numpy())
    
    x_0 = x_test[:,0]
    x_0[0] = 0
    x_0[1] = 120*np.pi/180
    u = 0*u_test[:,0]
    x_k[0] = x_0
    for t in range(200):
        x_kp1[t] = mlp(x_k[t],u).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest90.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_carttest120.npy',x_k.detach().cpu().numpy())
    
    # x_0 = x_test[:,0]
    # x_0[0] = 5*np.pi/180
    # u = 0*u_test[:,0]
    # x_k[0] = x_0
    # x_kpt[0] = x_0
    # for t in range(200):
    #     x_kp1[t] = mlp(x_k[t],u).flatten()
    #     x_k[t+1] = x_kp1[t]
        
    #     # x_kpt[t+1] = mlp(x_test[:,200+t],u)
    
    
    # # np.save('traj_xkp1_'+deg+'_pentest20.npy',x_kp1.detach().cpu().numpy())
    # np.save('traj_xk_'+deg+'_pentest5.npy',x_k.detach().cpu().numpy())
    
    # np.save('test1.npy',x_k.detach().cpu().numpy())
    # np.save('test2.npy',x_kpt.detach().cpu().numpy())
    # np.save('testgt.npy',x_test[:,200:400].detach().cpu().numpy())
    
    # x_0 = x_test[:,600]
    # # x_0[0] = 2*np.pi/180
    # u = 0*u_test[:,0]
    # x_k[0] = x_0
    # x_kpt[0] = x_0
    # for t in range(200):
    #     x_kp1[t] = mlp(x_k[t],u).flatten()
    #     x_k[t+1] = x_kp1[t]
        
    #     x_kpt[t+1] = mlp(x_test[:,600+t],u)
    
    
    # np.save('traj_xkp1_'+deg+'_pentest20.npy',x_kp1.detach().cpu().numpy())
    # np.save('traj_xk_'+deg+'_pentest2.npy',x_k.detach().cpu().numpy())
    
    # np.save('test21.npy',x_k.detach().cpu().numpy())
    # np.save('test22.npy',x_kpt.detach().cpu().numpy())
    # np.save('testgt2.npy',x_test[:,600:800].detach().cpu().numpy())
    
    # x_0 = x_test[:,1100]
    # # x_0[0] = 2*np.pi/180
    # u = 0*u_test[:,0]
    # x_k[0] = x_0
    # x_kpt[0] = x_0
    # for t in range(200):
    #     x_kp1[t] = mlp(x_k[t],u).flatten()
    #     x_k[t+1] = x_kp1[t]
        
    #     x_kpt[t+1] = mlp(x_test[:,1100+t],u)
    
    
    # # np.save('traj_xkp1_'+deg+'_pentest20.npy',x_kp1.detach().cpu().numpy())
    # # np.save('traj_xk_'+deg+'_pentest2.npy',x_k.detach().cpu().numpy())
    
    # np.save('test31.npy',x_k.detach().cpu().numpy())
    # np.save('test32.npy',x_kpt.detach().cpu().numpy())
    # np.save('testgt3.npy',x_test[:,1100:1300].detach().cpu().numpy())
    
    
    '''x_0 = x_test[:,0]
    x_0[0] = 5*np.pi/180
    u = 0*u_test[:,0]
    x_k[0] = x_0
    for t in range(200):
        x_kp1[t] = mlp(x_k[t]*max_x,u).flatten()
        x_k[t+1] = x_kp1[t]
    
    
    # np.save('traj_xkp1_'+deg+'_pentest10.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_'+deg+'_pentest5.npy',x_k.detach().cpu().numpy())'''
    
    # np.save('traj_y_'+deg+'_pen.npy',y_test.detach().cpu().numpy())
    print('Superposition Testing')
    # u1 = 0.25*torch.ones(1).to(device)
    # u2 = 0.15*torch.ones(1).to(device)
    y1 = mlp(0*x_test[:,0], u_test[:,0])#-x_test[:,0]
    y2 = mlp(0*x_test[:,0], u_test[:,1])#-x_test[:,0]
    ytotal = mlp(0*x_test[:,0], u_test[:,0]+u_test[:,1])#-x_test[:,0]
    print(y1)
    print(y2)
    print(ytotal)
    # sys.exit()
    
    print('Loss function for local model on local data:', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    # print('Loss function for local model on global data:', loss_function(mlp(x_test2,u_test2).to(device), y_test2.T))
    
    print('Actual Values')
    print(y_test[:,:10])
    # print(y_test2[:,:2])
    
    print('Predicted Values')
    print(x_kp1[:10].T)
    # print(mlp(x_test2[:,:2],u_test2[:,:2]))
    
    sys.exit()
    print('Traj Gen')
    # mlp = torch.load('MLP_pendulum_local_pm10')
    # mlp2 = torch.load('MLP_pendulum_local_pm10_2')
    
    x_0 = x_test
    u = u_test
    y_gt = y_test
    
    x_kp1 = mlp(x_0, u)
    
    np.save('traj_xkp1_45deg_pen.npy',x_kp1.detach().cpu().numpy())
    np.save('traj_xk_45deg_pen.npy',x_0.detach().cpu().numpy())
    np.save('traj_y_45deg_pen.npy',y_gt.detach().cpu().numpy())
    
    sys.exit()
    
    print('5 percent noise')
    eps = '5pc'
    np.save('training_loss_'+eps+'_cartpole_40.npy',x.detach().cpu().numpy())
    np.save('testing_loss_'+eps+'_cartpole_40.npy',x.detach().cpu().numpy())
    sys.exit()
    mlp = torch.load('MLP_cartpole_'+eps)
    mlp2 = torch.load('MLP_cartpole_'+eps+'_2')
    
    test = mlp(x_test, u_test)
    test2 = mlp2(x_test, u_test)
    t1 = loss_function(test, y_test.T)
    t2 = loss_function(test2, y_test.T)
    
    
    
    print('Training trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*10]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*10:30*11]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*10:30*11]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
    
    np.save('training_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_training_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Training Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*10:30*10+1]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*10:30*10+1]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*10:30*10+1]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*100:30*100+1]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*100:30*100+1]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*100:30*100+1]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*120:30*120+1]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*120:30*120+1]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*120:30*120+1]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    
    print('Total Testing error :')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*120:30*120+1]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*120:30*120+1]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*120:30*120+1]).float().to(device)
    
    x_o = np.zeros((4,30*4))
    x_kgt = np.zeros((4,30*4))
    x_kpred = np.zeros((4,30*4))
    for t in range(30):
        x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*0+t:30*0+t+1]).float().to(device)
        u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*0+t:30*0+t+1]).float().to(device)
        y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*0+t:30*0+t+1]).float().to(device)
        
        x_o[:,t] = x_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kgt[:,t] = y_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kpred[:,t] = mlp(x_test,u_test).detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        
        print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    
    for t in range(30):
        x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*100+t:30*100+t+1]).float().to(device)
        u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*100+t:30*100+t+1]).float().to(device)
        y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*100+t:30*100+t+1]).float().to(device)
        
        x_o[:,30+t] = x_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kgt[:,30+t] = y_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kpred[:,30+t] = mlp(x_test,u_test).detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
    for t in range(30):
        x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*400+t:30*400+t+1]).float().to(device)
        u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*400+t:30*400+t+1]).float().to(device)
        y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*400+t:30*400+t+1]).float().to(device)
        
        x_o[:,60+t] = x_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kgt[:,60+t] = y_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kpred[:,60+t] = mlp(x_test,u_test).detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
    for t in range(30):
        x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*800+t:30*800+t+1]).float().to(device)
        u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*800+t:30*800+t+1]).float().to(device)
        y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*800+t:30*800+t+1]).float().to(device)
        
        x_o[:,t+90] = x_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kgt[:,t+90] = y_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kpred[:,t+90] = mlp(x_test,u_test).detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
    # print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    # np.save('x_o_train.npy', x_o)
    # np.save('x_kgt_train.npy', x_kgt)
    # np.save('x_kpred_train.npy', x_kpred)
    
    # sys.exit()
    print('Testing trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*910]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*910:30*911]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*910:30*911]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
        
    print('%MSE Diff in Traj : ',100*loss_function(x.to(device), x_gt)/loss_function(0*x_gt, x_gt))
        
    np.save('testing_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_testing_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Testing Data (Single step predictions)')
    x_o = np.zeros((4,30))
    x_kgt = np.zeros((4,30))
    x_kpred = np.zeros((4,30))
    x_kpred2 = np.zeros((4,30))
    for t in range(30):
        x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*900+t:30*900+t+1]).float().to(device)
        u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*900+t:30*900+t+1]).float().to(device)
        y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*900+t:30*900+t+1]).float().to(device)
        
        x_o[:,t] = x_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kgt[:,t] = y_test.detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kpred[:,t] = mlp(x_test,u_test).detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        x_kpred2[:,t] = mlp2(x_test,u_test).detach().cpu().numpy().reshape(np.shape(x_o[:,t]))
        print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    # print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    np.save('x_o.npy', x_o)
    np.save('x_kgt.npy', x_kgt)
    np.save('x_kpred.npy', x_kpred)
    np.save('x_kpred2.npy', x_kpred2)
    sys.exit()
    
    
    print('10 percent noise')
    eps = '10pc'
    mlp = torch.load('MLP_cartpole_'+eps)
    
    print('Training trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*10]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*10:30*11]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*10:30*11]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
    
    np.save('training_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_training_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Training Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*10:30*11]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*10:30*11]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*10:30*11]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    
    print('Testing trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*910]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*910:30*911]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*910:30*911]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
        
    np.save('testing_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_testing_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Testing Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*910:30*911]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*910:30*911]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*910:30*911]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    
    
    
    print('20 percent noise')
    eps = '20pc'
    mlp = torch.load('MLP_cartpole_'+eps)
    
    print('Training trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*10]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*10:30*11]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*10:30*11]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
    
    np.save('training_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_training_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Training Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*10:30*11]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*10:30*11]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*10:30*11]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    
    print('Testing trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*910]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*910:30*911]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*910:30*911]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
        
    np.save('testing_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_testing_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Testing Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*910:30*911]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*910:30*911]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*910:30*911]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    
    
    print('50 percent noise')
    eps = '50pc'
    mlp = torch.load('MLP_cartpole_'+eps)
    
    print('Training trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*10]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*10:30*11]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*10:30*11]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
    
    np.save('training_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_training_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Training Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*10:30*11]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*10:30*11]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*10:30*11]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    
    print('Testing trajectory:')

    x_0 = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 30*910]).float().to(device)
    u = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:, 30*910:30*911]).float().to(device)
    y_gt = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:, 30*910:30*911]).float().to(device)
    
    x = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x_gt = 0*torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:, 0:31]).float().to(device)
    x[:,0] = x_0
    x_gt[:,0] = x_0
    
    for t in range(30):
        x[:,t+1] = mlp(x[:,t],u[:,t])
        x_gt[:,t+1] = y_gt[:,t]
        
    np.save('testing_traj_'+eps+'_cartpole.npy',x.detach().cpu().numpy())
    np.save('gt_testing_traj_'+eps+'_cartpole.npy',x_gt.detach().cpu().numpy())
    
    print('Testing Data (Single step predictions)')
    x_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_x.npy')[:,30*910:30*911]).float().to(device)
    u_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_u.npy')[:,30*910:30*911]).float().to(device)
    y_test = torch.from_numpy(np.load('eps'+eps+'_cartpoleNN_xkp1.npy')[:,30*910:30*911]).float().to(device)
    
    print('Loss function for '+eps+' model :', loss_function(mlp(x_test,u_test).to(device), y_test.T))
    print('Relative Loss function for '+eps+' model :', 100*loss_function(mlp(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
   
    
    '''mlp = torch.load('MLP_swimmer6_5pc')

    x_test = torch.from_numpy(np.load('eps20pc_swimmer6NN_x.npy')[:, 910]).float().to(device)
    u_test = torch.from_numpy(np.load('eps20pc_swimmer6NN_u.npy')[:, 910]).float().to(device)
    y_test = torch.from_numpy(np.load('eps20pc_swimmer6NN_xkp1.npy')[:, 910]).float().to(device)
    
    # x_test = torch.from_numpy(np.load('eps100pc_cartpoleNN_x.npy')[:, 910]).float().to(device)
    # u_test = torch.from_numpy(np.load('eps100pc_cartpoleNN_u.npy')[:, 910]).float().to(device)
    # y_test = torch.from_numpy(np.load('eps100pc_cartpoleNN_xkp1.npy')[:, 910]).float().to(device)

    y_pred = mlp(x_test/max_x, u_test/max_u).to(device)
    # print(x_test)
    # print(u_test)
    print('Predicted : ', y_pred)
    print('Actual : ', y_test)'''
    
    # x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:, 0]).float().to(device)
    # u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:, 0]).float().to(device)
    # y_test = torch.from_numpy(np.load('eps50pc_cartpoleNN_xkp1.npy')[:, 0]).float().to(device)

    # y_pred = mlp(x_test, u_test).to(device)
    # # print(x_test)
    # # print(u_test)
    # print('Predicted : ', y_pred)
    # print('Actual : ', y_test)

    # Training Error
    epsilon = 'low'
    mlplow = torch.load('MLP_cartpole_'+epsilon+'pc')
    
    epsilon = '5'
    mlp5 = torch.load('MLP_cartpole_'+epsilon+'pc')
    
    epsilon = '10'
    mlp10 = torch.load('MLP_cartpole_'+epsilon+'pc')
    
    epsilon = '20'
    mlp20 = torch.load('MLP_cartpole_'+epsilon+'pc')
    
    epsilon = '50'
    mlp50 = torch.load('MLP_cartpole_'+epsilon+'pc')
    
    epsilon = '100'
    mlp100 = torch.load('MLP_cartpole_'+epsilon+'pc')
    eps = 'eps'+epsilon
    TE = 0
    
    # print('Plotting trajectories : ')
    # x0 = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,:31]).float().to(device)
    # u = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,:30]).float().to(device)
    # x_gt = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,:31]).float().to(device)
    
    # xk = torch.zeros((x_gt.size()))
    # print(x0)
    # for t in range(30):
    
    
    '''print('TRAINING DATA')
    
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,30*10:30*11]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,30*10:30*11]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,30*10:30*11]).float().to(device)
    
    print('Loss function for low model :', loss_function(mlplow(x_test,u_test).to(device), y_test.T))
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    print('TESTING DATA')
    
    print('Loss function for low model :', loss_function(mlplow(x_test,u_test).to(device), y_test.T))
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,30*200:30*201]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,30*200:30*201]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,30*200:30*201]).float().to(device)
    
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    print('TRAIN')
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,100]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,100]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,100]).float().to(device)
    
    print('Loss function for low model :', loss_function(mlplow(x_test,u_test).to(device), y_test.T))
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    print('Rel Loss function for e5 model :', 100*loss_function(mlp5(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e10 model :', 100*loss_function(mlp10(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e20 model :', 100*loss_function(mlp20(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e50 model :', 100*loss_function(mlp50(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e100 model :', 100*loss_function(mlp100(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    print('Test')
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,1500]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,1500]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,1500]).float().to(device)
    
    print('Loss function for low model :', loss_function(mlplow(x_test,u_test).to(device), y_test.T))
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    print('Rel Loss function for e5 model :', 100*loss_function(mlp5(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e10 model :', 100*loss_function(mlp10(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e20 model :', 100*loss_function(mlp20(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e50 model :', 100*loss_function(mlp50(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e100 model :', 100*loss_function(mlp100(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    print('TEst single point 2')
    
    print('Loss function for low model :', loss_function(mlplow(x_test,u_test).to(device), y_test.T))
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,1600]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,1600]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,1600]).float().to(device)
    
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    print('Rel Loss function for e5 model :', 100*loss_function(mlp5(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e10 model :', 100*loss_function(mlp10(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e20 model :', 100*loss_function(mlp20(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e50 model :', 100*loss_function(mlp50(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e100 model :', 100*loss_function(mlp100(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('TEst single point 3')
    
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,1700]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,1700]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,1700]).float().to(device)
    
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    print('Rel Loss function for e5 model :', 100*loss_function(mlp5(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e10 model :', 100*loss_function(mlp10(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e20 model :', 100*loss_function(mlp20(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e50 model :', 100*loss_function(mlp50(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e100 model :', 100*loss_function(mlp100(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('TEst single point 4')
    
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,1800]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,1800]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,1800]).float().to(device)
    
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    
    print('Rel Loss function for e5 model :', 100*loss_function(mlp5(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e10 model :', 100*loss_function(mlp10(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e20 model :', 100*loss_function(mlp20(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e50 model :', 100*loss_function(mlp50(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    print('Rel Loss function for e100 model :', 100*loss_function(mlp100(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    print('TEst single point 5')
    
    x_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_x.npy')[:,16000]).float().to(device)
    u_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_u.npy')[:,16000]).float().to(device)
    y_test = torch.from_numpy(np.load('eps5pc_cartpoleNN_xkp1.npy')[:,16000]).float().to(device)
    
    print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))'''
    
    # print('TESTING OPTIMAL TRAJECTORY')
    
    # x_test = torch.from_numpy(np.load('opt_cartpoleNN_x.npy')).float().to(device)
    # u_test = torch.from_numpy(np.load('opt_cartpoleNN_u.npy')).float().to(device)
    # y_test = torch.from_numpy(np.load('opt_cartpoleNN_xkp1.npy')).float().to(device)
    
    # print('Loss function for e5 model :', loss_function(mlp5(x_test,u_test).to(device), y_test.T))
    # print('Loss function for e10 model :', loss_function(mlp10(x_test,u_test).to(device), y_test.T))
    # print('Loss function for e20 model :', loss_function(mlp20(x_test,u_test).to(device), y_test.T))
    # print('Loss function for e50 model :', loss_function(mlp50(x_test,u_test).to(device), y_test.T))
    # print('Loss function for e100 model :', loss_function(mlp100(x_test,u_test).to(device), y_test.T))
    
    # print('Rel Loss function for e5 model :', 100*loss_function(mlp5(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    # print('Rel Loss function for e10 model :', 100*loss_function(mlp10(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    # print('Rel Loss function for e20 model :', 100*loss_function(mlp20(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    # print('Rel Loss function for e50 model :', 100*loss_function(mlp50(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    # print('Rel Loss function for e100 model :', 100*loss_function(mlp100(x_test,u_test).to(device), y_test.T)/loss_function(0*y_test.T, y_test.T))
    
    '''TE_per_traj = 0
    for i in range(900):
        x_test = torch.from_numpy(np.load(eps+'pc_swimmer3NN_x.npy')[:, i]).float().to(device)
        u_test = torch.from_numpy(np.load(eps+'pc_swimmer3NN_u.npy')[:, i]).float().to(device)
        y_test = torch.from_numpy(np.load(eps+'pc_swimmer3NN_xkp1.npy')[:, i]).float().to(device)
        TE+=loss_function(mlp(x_test,u_test).to(device), y_test.T)
    TE_per_traj = TE/900
    print('Training Error: ', TE)
    print('Training Error per traj: ', TE_per_traj)
    
    # Testing Error
    test = 0
    test_per_traj = 0
    for i in range(100):
        x_test = torch.from_numpy(np.load(eps+'pc_swimmer3NN_x.npy')[:, 900+i]).float().to(device)
        u_test = torch.from_numpy(np.load(eps+'pc_swimmer3NN_u.npy')[:, 900+i]).float().to(device)
        y_test = torch.from_numpy(np.load(eps+'pc_swimmer3NN_xkp1.npy')[:, 900+i]).float().to(device)
        test+=loss_function(mlp(x_test,u_test).to(device), y_test.T)
    test_per_traj = test/100
    print('Testing Error: ', test)
    print('Testing Error per traj: ', test_per_traj)'''