import torch
from model import HierarchialVAE, DiffusionProcess, Denoise_net, Encoder_Block, Decoder_Block
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial
from dataset import CreateInOutSequence
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


batch_size = 16
sequence_length = 12
prediction_length = 5
VAE = HierarchialVAE(Encoder_Block = Encoder_Block, Decoder_Block = Decoder_Block , latent_dim2 = 5, latent_dim1 = 2, feature_size2 = 36, 
                 feature_size1 = 9, hidden_size = 2, pred_length = 5, num_features = 12, seq_length = 12)
Diffusion_Process = DiffusionProcess(num_diff_steps = 10, vae = VAE, beta_start = 0.01, beta_end = 0.1, scale = 0.5)
Denoise_Net = Denoise_net(in_channels = 16,dim = 16, size = 5)
criterion=nn.MSELoss()
optimizer1=optim.Adam(VAE.parameters(),lr=3e-3)
optimizer2=optim.Adam(Denoise_Net.parameters(),lr=3e-3)
scheduler1= StepLR(optimizer1, step_size=2, gamma=0.5)
scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.5)

def train(epochs,train_dataloader,val_dataloader,VAE,Diffusion,num_diff_steps):
    
    train_loss=[]
    val_loss=[]
    
    for epoch in range(0,epochs):
        
        total_loss=0
        VAE.train()
        Diffusion.train()
        for i,(x,y) in enumerate(train_dataloader):
            if(x.size(0)!=16):
                break
            vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
            diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
            
            for time in range(1,num_diff_steps + 1):
                output, y_noisy = Diffusion_Process.diffuse(x,y,time)
                vae_out[:,:,time-1] = output
                diff_out[:,:,time-1] = y_noisy 
            mean_vae = torch.mean(vae_out, dim = 2)
            mean_diff = torch.mean(diff_out, dim = 2)
            var_vae = torch.std(vae_out, dim = 2)
            var_diff = torch.std(diff_out, dim = 2)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            mse_loss = criterion(mean_vae, mean_diff)
            term1 = (mean_vae - mean_diff) / var_diff
            term2 = var_vae / var_diff
            kl_loss =  0.5 * ((term1 * term1).sum() + (term2 * term2).sum()) - 40 - torch.log(term2).sum()
            kl_loss = kl_loss.sum()
            
            ran=torch.randint(low=1,high=num_diff_steps + 1,size=(1,))
            y_nn=vae_out[:,:,:]
            
            E = Denoise_Net(y_nn).sum()
            grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0] 
            dsm_loss = torch.mean(torch.sum((y.unsqueeze(2)-y_nn+grad_x*1)**2, [0,1,2])).float()
            loss = 4*mse_loss+0.01*kl_loss+ 0.1*dsm_loss
            total_loss+=loss
            loss.backward()
            optimizer1.step()
            optimizer2.step()
        
        scheduler1.step()
        scheduler2.step()
        totalval_loss=0
        
        VAE.eval()
        Diffusion.eval()
        for i,(x,y) in enumerate(val_dataloader):
            if(x.size(0)!=16):
                break
            vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
            diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
                
            for time in range(1,num_diff_steps + 1):
                output, y_noisy = Diffusion_Process.diffuse(x,y,time)
                vae_out[:,:,time-1] = output
                diff_out[:,:,time-1] = y_noisy 
            mean_vae = torch.mean(vae_out, dim = 2)
            mean_diff = torch.mean(diff_out, dim = 2)
            var_vae = torch.std(vae_out, dim = 2)
            var_diff = torch.std(diff_out, dim = 2)
            mse_loss = criterion(mean_vae, mean_diff)
            term1 = (mean_vae - mean_diff) / var_diff
            term2 = var_vae / var_diff
            kl_loss =  0.5 * ((term1 * term1).sum() + (term2 * term2).sum()) - 40 - torch.log(term2).sum()
            kl_loss = kl_loss.sum()
            ran=torch.randint(low=1,high=num_diff_steps + 1,size=(1,))
            y_nn=vae_out[:,:,:]
            E = Denoise_Net(y_nn).sum()
            grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0] 
            dsm_loss = torch.mean(torch.sum((y.unsqueeze(2)-y_nn+grad_x*1)**2, [0,1,2])).float()
            valloss = 4*mse_loss+0.01*kl_loss+ 0.1*dsm_loss
            totalval_loss+=valloss
        
        train_loss.append(total_loss/(len(train_dataloader)))
        val_loss.append(totalval_loss/(len(val_dataloader)))
        print(f"Epoch: {epoch+1}")
        print(f"Training :: Loss:{train_loss[epoch]}")
        print(f"Validation :: Loss:{val_loss[epoch]}")
        
    return train_loss,val_loss

def test(test_dataloader,VAE,Diffusion,num_diff_steps):
    totaltest_loss=0
    totalmse_loss=0
    totalkl_loss=0
    totaldsm_loss=0
    predicted_seq=[]
    inp_seq=[]
    target_seq=[]
    for i,(x,y) in enumerate(test_dataloader):
        if(x.size(0)!=16):
            break
        vae_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))
        diff_out = torch.zeros((y.size(0), y.size(1),num_diff_steps))

        for time in range(1,num_diff_steps + 1):
            output, y_noisy = Diffusion_Process.diffuse(x,y,time)
            vae_out[:,:,time-1] = output
            diff_out[:,:,time-1] = y_noisy 
        mean_vae = torch.mean(vae_out, dim = 2)
        mean_diff = torch.mean(diff_out, dim = 2)
        var_vae = torch.std(vae_out, dim = 2)
        var_diff = torch.std(diff_out, dim = 2)
        mse_loss = criterion(mean_vae, mean_diff)
        term1 = (mean_vae - mean_diff) / var_diff
        term2 = var_vae / var_diff
        kl_loss =  0.5 * ((term1 * term1).sum() + (term2 * term2).sum()) - 40 - torch.log(term2).sum()
        kl_loss = kl_loss.sum()
        ran=torch.randint(low=1,high=num_diff_steps + 1,size=(1,))
        y_nn=vae_out[:,:,:]
        E = Denoise_Net(y_nn).sum()
        grad_x = torch.autograd.grad(E, y_nn, create_graph=True)[0] 
        dsm_loss = torch.mean(torch.sum((y.unsqueeze(2)-y_nn+grad_x*1)**2, [0,1,2])).float()
        testloss = 4*mse_loss+0.01*kl_loss+ 0.1*dsm_loss
        totalmse_loss+=4*mse_loss
        totalkl_loss+=0.01*kl_loss
        totaldsm_loss+=0.1*dsm_loss
        totaltest_loss+=testloss
        inp_seq.append(x)
        predicted_seq.append(mean_vae - 0.1*torch.mean(grad_x,dim=2))
        target_seq.append(y)
    avg_test_loss=totaltest_loss/(len(test_dataloader))
    avg_mse_loss=totalmse_loss/(len(test_dataloader))
    avg_kl_loss=totalkl_loss/(len(test_dataloader))
    avg_dsm_loss=totaldsm_loss/(len(test_dataloader))
    print(f"Test total Loss : {avg_test_loss}")
    print(f"Test MSE Loss : {avg_mse_loss}")
    print(f"Test KL Loss : {avg_kl_loss}")
    print(f"Test DSM Loss : {avg_dsm_loss}")
    return inp_seq,predicted_seq,target_seq,avg_test_loss,avg_mse_loss,avg_kl_loss,avg_dsm_loss

stock_df = pd.read_csv('final_data.csv')
sequenced_data = CreateInOutSequence(stock_df,sequence_length,prediction_length)

def train_test_splitter(dataset, split = 0.8):
    indices = list(range(len(dataset)))

    train_indices, test_indices = train_test_split(indices, train_size=split, shuffle=False)
    val_indices, test_indices = train_test_split(test_indices, train_size=0.5, shuffle=False)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset= torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    train_size=len(train_dataset)
    test_size=len(val_dataset)
    val_size=len(test_dataset)
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size

train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = train_test_splitter(sequenced_data, split = 0.8)
train_dataloader=DataLoader(train_dataset,batch_size=16,shuffle=False)
val_dataloader=DataLoader(val_dataset,batch_size=16,shuffle=False)
test_dataloader=DataLoader(test_dataset,batch_size=16,shuffle=False)
entire_dataloader=DataLoader(sequenced_data,batch_size=16,shuffle=False)

train_loss,val_loss = train(epochs =20
                             ,train_dataloader = train_dataloader, val_dataloader = val_dataloader, VAE = VAE,Diffusion = Denoise_Net, num_diff_steps = 10)

train_loss = [tensor.detach() for tensor in train_loss]
val_loss = [tensor.detach() for tensor in val_loss]

plt.figure(figsize=(11, 8))
plt.plot((np.arange(2,21,1)),train_loss[1:],label='Validation Loss')
plt.plot((np.arange(2,21,1)),val_loss[1:],label='Training Loss')
plt.title("Loss vs Epochs",fontsize=16)
plt.xlabel('Epochs')
plt.ylabel("Loss (MSE+KL+DSM)",fontsize=16)
plt.legend()
plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.grid()
plt.savefig('Loss vs Epochs.png')

_,_,_,loss,mse,kl,dsm = test(test_dataloader,VAE,Denoise_net,10)

inp,pred,tar,_,_,_,_ = test(entire_dataloader,VAE,Denoise_net,10)

target_sequence = [item for sublist in tar for item in sublist]
pred_sequence = [item for sublist in pred for item in sublist]

tarcont_seq = [target_sequence[i] for i in range(len(target_sequence)) if i % 5 == 0]
predcont_seq = [pred_sequence[i] for i in range(len(pred_sequence)) if i % 5 == 0]

denorm_tar=stock_df['Close'][6+5:(len(tarcont_seq) + 11)]*tarcont_seq
denorm_pred=stock_df['Close'][6+5:(len(predcont_seq) + 11)]*predcont_seq

plt.figure(figsize=(14,8))
plt.plot(np.arange(0,len(denorm_tar[0:2048]),1),denorm_tar[0:2048],label='Target')
plt.plot(np.arange(0,len(denorm_pred[0:2048]),1),denorm_pred[0:2048],label='Predicted')
plt.grid()
plt.legend()
plt.title('Actual vs Predicted Stock Price (Training Data)',fontsize=16)
plt.xlabel('Days',fontsize=16)
plt.ylabel('Stock Price (Denormalized)',fontsize=16)
plt.savefig('Training.png')

plt.figure(figsize=(14,8))
plt.plot(np.arange(0,len(denorm_tar[2048:2048+256]),1),denorm_tar[2048:2048+256],label='Target')
plt.plot(np.arange(0,len(denorm_pred[2048:2048+256]),1),denorm_pred[2048:2048+256],label='Predicted')
plt.grid()
plt.legend()
plt.title('Actual vs Predicted Stock Price (Validation Data)',fontsize=16)
plt.xlabel('Days',fontsize=16)
plt.ylabel('Stock Price (Denormalized)',fontsize=16)
plt.savefig('Validation.png')

plt.figure(figsize=(14,8))
plt.plot(np.arange(0,len(denorm_tar[2048+256:]),1),denorm_tar[2048+256:],label='Target')
plt.plot(np.arange(0,len(denorm_pred[2048+256:]),1),denorm_pred[2048+256:],label='Predicted')
plt.grid()
plt.legend()
plt.title('Actual vs Predicted Stock Price (Test Data)',fontsize=16)
plt.xlabel('Days',fontsize=16)
plt.ylabel('Stock Price (Denormalized)',fontsize=16)
plt.savefig('Testing.png')


mape_train = np.mean(np.abs((denorm_tar[0:2048] - denorm_pred[0:2048]) / denorm_tar[0:2048])) * 100
mape_val = np.mean(np.abs((denorm_tar[2048:2048+256] - denorm_pred[2048:2048+256]) / denorm_tar[2048:2048+256])) * 100
mape_test = np.mean(np.abs((denorm_tar[2048+256:] - denorm_pred[2048+256:]) / denorm_tar[2048+256:])) * 100
rmse_train = np.sqrt(mean_squared_error(denorm_tar[0:2048], denorm_pred[0:2048]))
rmse_val = np.sqrt(mean_squared_error(denorm_tar[2048:2048+256], denorm_pred[2048:2048+256]))
rmse_test = np.sqrt(mean_squared_error(denorm_tar[2048+256:], denorm_pred[2048+256:]))

print(f'MAPE for Training Data: {mape_train}%')
print(f'MAPE for Validation Data: {mape_val}%')
print(f'MAPE for Test Data: {mape_test}%')
print(f'RMSE for Training Data: {rmse_train}')
print(f'RMSE for Validation Data: {rmse_val}')
print(f'RMSE for Test Data: {rmse_test}')
