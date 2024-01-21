import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from functools import partial
import numpy as np

class DiffusionProcess(nn.Module):
    def __init__(self, num_diff_steps, vae, beta_start, beta_end, scale):
        super().__init__()
        to_torch = partial(torch.tensor, dtype = torch.float32)
        ## Initializing variables like number of time stamps, the Hierarchial VAE to make predictions, start and end values
        ## for beta, which governs the variance schedule
        self.num_diff_steps = num_diff_steps
        self.vae = vae
        self.beta_start = beta_start
        self.beta_end = beta_end
        ## Defining a linearly varying variance schedule for the conditional noise at every timestamp 
        betas = np.linspace(beta_start, beta_end,  num_diff_steps, dtype = np.float32)
        
        ## Performing reparametrization to calculate output at time t directly using x_start
        alphas = 1 - betas
        alphas_target = 1 - betas*scale
        ## Computing the cumulative product for the input as well as output noise schedule
        alphas_cumprod = np.cumprod(alphas, axis = 0)
        alphas_target_cumprod = np.cumprod(alphas_target, axis = 0)
        
        ## Converting all the computed quantities to tensors and detaching them from the computation graph (setting requires_grad to False)
        betas = torch.tensor(betas, requires_grad = False)
        alphas_cumprod = torch.tensor(alphas_cumprod, requires_grad = False)
        alphas_target_cumprod = torch.tensor(alphas_target_cumprod, requires_grad = False)
        
        ## Computing scaling factors for mean and variance respectively
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).detach().requires_grad_(False)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).detach().requires_grad_(False)
        self.sqrt_alphas_target_cumprod = torch.sqrt(alphas_target_cumprod).detach().requires_grad_(False)
        self.sqrt_one_minus_alphas_target_cumprod = torch.sqrt(1 - alphas_target_cumprod).detach().requires_grad_(False)
        
    ## Defining the forward pass
    def diffuse(self, x_start, y_target, timestamp):
        ## Generating a random noise vector sampled from a standard normal of the size x_start and y_target respectively
        noise = torch.randn_like(x_start)
        noise_target = torch.randn_like(y_target)
        
        ## Computing the sampled value using the reparametrization trick and using that to calculate x_noisy and y_noisy
        x_noisy = self.sqrt_alphas_cumprod[timestamp - 1]*x_start + self.sqrt_one_minus_alphas_cumprod[timestamp - 1]*noise
        y_noisy = self.sqrt_alphas_target_cumprod[timestamp - 1]*y_target + self.sqrt_one_minus_alphas_target_cumprod[timestamp - 1]*noise_target
    
        ## Performing a forward pass through the Hierarchial VAE to generate noisy predictions
        output = self.vae(x_noisy)
        return output, y_noisy
    
import torch.nn.init as init

## Initializing weights using Xavier Initialization
def init_weights(layer):
    init.xavier_uniform_(layer.weight)
    layer_name=layer._class.name_
    if layer.find("Conv")!=-1:
        layer.weight.data.normal_(0.0,0.25)
    elif layer.find("BatchNorm")!=-1:
        layer.weight.data.normal(1.00,0.25)
        layer.bias.data.fill_(0.00)

## Defining a custom Conv2D class with the padding size such that the input size and output size remain the same
class Conv2D(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride):
        super(Conv2D,self).__init__()
        ## Required padding size = kernel_size - 1/2
        padding=int((kernel_size-1)/2)
        self.layer=nn.Conv2d(input_dim,output_dim,kernel_size,stride=stride,padding=padding,bias=True)
    ## Performing the forward pass
    def forward(self,input):
        return self.layer(input)

## Defining the module for Swish Activation or Sigmoid Linear Unit
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        self.layer=nn.SiLU()
    def forward(self,input):
        return self.layer(input)

## Performing Batch Normalization by inherting it from torch.nn
class BatchNorm(nn.Module):
    def __init__(self,batch_dim,size):
        super(BatchNorm,self).__init__()
        ## Equivalent to BatchNorm as first dimension is batch_size
        self.layer=nn.LayerNorm([batch_dim,size,size])

    def forward(self,input):
        return self.layer(input)
        
class SE(nn.Module):
    def __init__(self,channels_in,channels_out):
        super(SE,self).__init__()
        ## Defining number of units to be compressed into
        num_hidden=max(channels_out//16,4)
        
        ## Defining the network which compresses and expands to focus on features rather than noise
        ## 2 networks req as 2 different input output dimensions are present in the Hierarchial VAE
        self.se=nn.Sequential(nn.Linear(1,num_hidden),nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, 144), nn.Sigmoid())
        self.se2=nn.Sequential(nn.Linear(1,num_hidden),nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, 36), nn.Sigmoid())

    def forward(self,input):
            
        ## Getting compressed vector
        se=torch.mean(input,dim=[1,2])
        ## Flattening out the layer
        se=se.view(se.size(0),-1)
        
        if(input.size(1)==12):
            se=self.se(se)
            se=se.view(se.size(0),12,12)
        else:
            se=self.se2(se)
            se=se.view(se.size(0),6,6)
        ## Returning appropriate mapped feature
        return input*se

## Performing pooling for downsampling using nn.AvgPool2D and using a kernel of size 2 to ensure that output size is halved
class Pooling(nn.Module):
    def __init__(self):
        super(Pooling,self).__init__()
        ## Using a 2x2 kernel and a stride of 2 in both directions
        self.mean_pool = nn.AvgPool2d(kernel_size=(2, 2),padding=0,stride=(2,2))
    def forward(self,input):
        return self.mean_pool(input)

## Defining a class to compute square of a quantity
class Square(nn.Module):
    def __init__(self):
        super(Square,self).__init__()
        pass
    def forward(self,input):
        return input**2
    
## Defining the encoder block to be used in Hierarchial VAE to convert to input into its latent space representation
class Encoder_Block(nn.Module):
    def __init__(self,input_dim,size,output_dim):
        super().__init__()
        ## Initializing the in and out dimensions of the conv layers and SE block
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = size
        ## Defining the encoder layers i.e 2 Conv2D layers followed by Batch Normalization, a Conv2D layer of kernel size 1 and Squeeze and excitation at the end
        self.seq=nn.Sequential(Conv2D(input_dim,input_dim,kernel_size=5,stride=1),
                               Conv2D(input_dim,input_dim,kernel_size=1,stride=1),
                               BatchNorm(input_dim,size),Swish(),
                               Conv2D(input_dim,input_dim,kernel_size=3,stride=1),
                               SE(input_dim,output_dim))
    def forward(self,input):
        ## Computing the final output as the sum of scaled encoded output and original input (result of skip connection i.e. residual encoder)
        return input +0.1*self.seq(input)
    
## Defining the decoder to be used in Hierarchial VAE to convert from latent space representations to noisy outputs 
class Decoder_Block(nn.Module):
    def __init__(self,dim,size,output_dim):
        super().__init__()
        ## Defining the decoder net which comprises of Conv2D layers, BatchNorm and SE Blocks
        ## We ensure that the dimension of the input and output stays the same at all instants as down/up sampling is done in a separate block 
        self.seq = nn.Sequential(
            BatchNorm(dim,size),
            Conv2D(dim,dim,kernel_size=1,stride=1),
            BatchNorm(dim,size), Swish(),
            Conv2D(dim,dim, kernel_size=5, stride=1),
            BatchNorm(dim,size), Swish(),
            Conv2D(dim, dim, kernel_size=1, stride = 1),
            BatchNorm(dim,size),
            ## SE Block just compresses and expands which allows it to ignore noise and focus on actual indicators
            SE(dim,output_dim))
    ## Computing the final output similar to encoder taking into account the skip connection
    def forward(self,input):
        return input+0.1*self.seq(input)
    
## Defining the class for the Hierarchial VAE which takes as input various hyperparameters and the classes for encoder and decoder blocks
class HierarchialVAE(nn.Module):
    def __init__(self, Encoder_Block, Decoder_Block, latent_dim2 = 5, latent_dim1 = 2, feature_size2 = 36, 
                 feature_size1 = 9, hidden_size = 2, pred_length = 5, num_features = 12, seq_length = 12, batch_size = 16):
        super().__init__()
        ## Initializing the encoder at the beginning when x_start has 12 features
        self.Encoder1 = Encoder_Block(input_dim = batch_size, output_dim = batch_size, size = 12)
        ## Initializing the encoder reqd after downsampling when input has 6 features 
        self.Encoder2 = Encoder_Block(input_dim = batch_size, output_dim = batch_size, size = 6)
        ## Initializing the decoder reqd after upsampling which gives y_noisy at the output
        self.Decoder1 = Decoder_Block(dim = batch_size,size = 12,output_dim = batch_size)
        ## Initializing the first decoder which obtains an input of size batchx6x6
        self.Decoder2 = Decoder_Block(dim = batch_size,size = 6,output_dim = batch_size)
        
        ## Initializing dimensions of both latent vectors, feature size of both the intermediate feature maps 
        self.latent_dim2 = latent_dim2
        self.latent_dim1 = latent_dim1
        self.feature_size2 = feature_size2
        self.feature_size1 = feature_size1
        ## Initializing the initial hidden state with a tensor of zeros with dimension equal to that of the final latent vector
        self.hidden_size = hidden_size
        self.hidden_state = torch.zeros(self.latent_dim1)
        ## Initializing batch_size
        self.batch_size= batch_size
        
        ## Defining the upsampling blocks required at 2 different stages in the entire network (2 networks reqd as size of input feature map varies throughout the network)
        self.upsample1 = nn.Upsample(size=(6, 6), mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(size=(12, 12), mode='bilinear', align_corners=False)
        ## Defining linear layers that map flattened feature maps to latent space dimensions and vice versa
        self.fc12 = nn.Linear(feature_size2,2*latent_dim2)
        self.fc11 = nn.Linear(feature_size1,2*latent_dim1)
        self.fc22 = nn.Linear(latent_dim2, feature_size2)
        self.fc21 = nn.Linear(latent_dim1, feature_size1)
        ## Defining pooling layer for downsampling
        self.mean_pool = nn.AvgPool2d(kernel_size=(2, 2),padding=0,stride=(2,2))
        ## The final linear layer which maps the VAE output to the output dimension
        self.fc_final = nn.Linear(num_features*seq_length, pred_length)
        
    
    def forward(self,x_start):
        ## We pass the input through two encoder blocks followed by pooling which reduces the feature map size to 6x6
        out = self.Encoder1(x_start)
        out = self.Encoder1(out)
        out = self.mean_pool(out)
        ## Reshaping the feature map and storing as it is required for sampling 
        feature_map2 = out.view(out.size(0),6,6)
        ## Encoding and Pooling the output once again which reduces the feature map size to 3x3 
        out = self.Encoder2(out)
        out = self.mean_pool(out)
        ## Flattening the final feature map and passing it through the linear layer which maps it to a latent vector of 
        ## dimension 4 (latent vector is dimension 2, but we predict both the mean and variances)
        feature_map1 = out.view(out.size(0),-1)
        z1 = self.fc11(feature_map1)
        ## Randomly sampling noise from a standard normal
        noise1 = torch.randn((out.size(0),self.latent_dim1))
        ## Applying the reparametrization trick to get the sampled value
        sampled_z1 = self.reparametrize(noise1,z1)
        ## Adding the initial hidden vector to the sampled output and converting it back to 3x3 feature map using a linear layer
        out = sampled_z1 + self.hidden_state
        out = self.fc21(out)
        out = out.view(out.size(0),3,3)
        ## Upsampling to dimension 6x6
        out = self.upsample1(out.unsqueeze(0)).squeeze(0)
        ## Passing it through the decoder and combining it with feature map 2 to sample from the 2nd latent vector
        out = self.Decoder2(out)
        ## Maps to a dimension of 10 after flattening the vector which means means and variances of a latent vector of dim = 5
        z_decoder = (feature_map2 + out).view(out.size(0),-1)
        z2 = self.fc12(z_decoder)
        ## In a similar fashion, we get the sampled value from z2
        noise2 = torch.randn((out.size(0),self.latent_dim2))
        sampled_z2 = self.reparametrize(noise2,z2)
        ## We convert it back to dim = 36 using a linear layer followed by reshaping it to 6x6
        z2_upsampled = self.fc22(sampled_z2).view(out.size(0),6,6)
        ## Upsampling to the original dimension of 12x12
        out = out + z2_upsampled
        out = self.upsample2(out.unsqueeze(0)).squeeze(0)
        out = self.Decoder1(out)
        out = self.Decoder1(out)
        ## Passing it through the final linear layer to map it to the shape of output
        out = self.fc_final(out.view(out.size(0),-1))
        return out
        
    def reparametrize(self,noise,z):
        ## Getting the batch_size
        zsize=int(z.size(1))
        ## Initializing tensors for mean and variances
        sampled_z = torch.zeros((noise.size(0),zsize//2))
        mu=torch.zeros((noise.size(0),zsize//2))
        sig=torch.zeros((noise.size(0),zsize//2))
        for i in range(0,zsize//2):
            mu[:,i]=z[:,i]
            sig[:,i]=z[:,zsize//2+i]
            ## Computing the sampled value
            sampled_z[:,i]=mu[:,i] + noise[:,i]*sig[:,i]
        return sampled_z

## Defining the network for denoising score matching 
class Denoise_net(nn.Module):
    def __init__(self,in_channels,dim,size,number=5):
        super().__init__()
        ## 2*number is number of diffusion samples used for denoise calculation
        ## Initializing the input dimension (actually prediction length in this case)
        hw = size
        self.dim=dim
        ## Number of input channels (batched mapping)
        self.channels=in_channels
        ## Defining the network for energy calculation
        self.conv=Conv2D(self.channels,dim,3,1)
        self.conv1=Conv2D(dim,dim,3,1)
        self.relu1=nn.ELU()
        self.pool1=Pooling()
        self.conv2=Conv2D(dim,dim,3,1)
        self.relu2=nn.ELU()
        self.conv3=Conv2D(dim,dim,3,1)
        self.relu3=nn.ELU()
        ## Getting interaction energy and self energy component field terms
        self.f1=nn.Linear((int(hw/2)*number),1)
        self.f2=nn.Linear((int(hw/2)*number),1)
        self.fq=nn.Linear((int(hw/2)*number),1)
        self.square=Square()
    
    def forward(self,input):
        output=self.conv(input)
        output1=self.conv1(output)
        output2=self.relu1(output1)
        ## Resnet type output computation for stable gradient flow
        output2=output2+output1
        ## Pooling to increase the receptive field 
        output3=self.pool1(output2)
        output4=self.conv2(output3)
        output5=self.relu2(output4)
        output5=output5+output4
        output7=self.conv3(output5)
        output8=self.relu3(output7)
        l1=self.f1(output8.view(input.size(0),-1))
        l2=self.f2(output8.view(input.size(0),-1))
        lq=self.fq(self.square(output8.view(input.size(0),-1)))
        ## Getting gradient of energy term per sample (gradient of energy term is what we are concerned with)
        out=l1*l2 +lq
        out=out.view(-1)
        return out

