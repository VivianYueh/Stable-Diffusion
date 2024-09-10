import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

#定義img_transform
IMG_SIZE = 64
BATCH_SIZE = 128

device = "cpu"
print(device)

img_transform = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Scales data into [0,1]
    transforms.Lambda(lambda x: x.to(device)),
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
img_transform = transforms.Compose(img_transform) 

pic_path = 'D:\python\diffusion_model\datasets'
num_samples = 8

data = torchvision.datasets.ImageFolder(root=pic_path, transform=img_transform)
plt.figure(figsize=(10,10))

'''for i,img in enumerate(data):
  if i == num_samples:
    break
  plt.subplot(int(num_samples/4) + 1, 4, i + 1)
  plt.imshow(torch.permute(img[0], (1,2,0)))'''

#plt.tight_layout()
#plt.show()
def forward_process(x_0, t):
    noise = torch.randn_like(x_0) #回傳與X_0相同size的noise tensor，也就是reparameterization的epsilon
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_oneminus_alphas_cumprod_t = sqrt_oneminus_alphas_cumprod[t]
    
    return sqrt_alphas_cumprod_t*x_0 + sqrt_oneminus_alphas_cumprod_t*noise, noise

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_oneminus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    #element-wise的運算
    return sqrt_alphas_cumprod_t*x_0 + sqrt_oneminus_alphas_cumprod_t*noise, noise

def linear_beta_schedule(timesteps=500, start=0.0001, end=0.02):
    '''
    return a tensor of a linear schedule
    '''
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def alpha_cumprod_cal(betas):
  alphas = 1-betas
  return torch.cumprod(alphas, dim=0)

#precalculations
T = 300
lin_betas = linear_beta_schedule(timesteps=T)
cos_betas = cosine_beta_schedule(timesteps=T)
qud_betas = quadratic_beta_schedule(timesteps=T)
sig_betas = sigmoid_beta_schedule(timesteps=T)
#alphas = 1-lin_betas

plt.plot(alpha_cumprod_cal(lin_betas),label='linear')
plt.plot(alpha_cumprod_cal(cos_betas),label='cos')
plt.plot(alpha_cumprod_cal(qud_betas),label='quadratic')
plt.plot(alpha_cumprod_cal(sig_betas),label='sigmoid')
plt.xlabel('timesteps',{'fontsize':20,'color':'black'})    # 設定 x 軸標籤
plt.ylabel('alpha',{'fontsize':20,'color':'black'})  # 設定 y 軸標籤
plt.legend(
    loc='best',
    fontsize=20,
    shadow=False,
    facecolor='white',
    edgecolor='#000')
plt.show()

'''alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_oneminus_alphas_cumprod = torch.sqrt(1-alphas_cumprod)

import numpy as np
# Simulate forward diffusion
image = next(iter(data))[0]

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

for idx in range(0, T, stepsize):
    t = idx
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    image, noise = forward_process(image, t)
    show_tensor_image(image)
plt.tight_layout()
plt.show()'''

import numpy as np
# Simulate forward diffusion

plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    
subidx = 1

for i in [lin_betas,cos_betas,qud_betas,sig_betas]:
  alphas=1-i
  alphas_cumprod = torch.cumprod(alphas, dim=0)
  sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
  sqrt_oneminus_alphas_cumprod = torch.sqrt(1-alphas_cumprod)
  image = next(iter(data))[0]


  for idx in range(0, T, stepsize):
    t = idx
    plt.subplot(4, num_images, subidx)
    image, noise = forward_process(image, t)
    subidx+=1
    show_tensor_image(image)
    plt.title(f'Time Step:{idx}')

  #plt.suptitle(title[title_idx])
  #plt.tight_layout()
  #plt.savefig(f'D:\python\diffusion_model\images\{title[title_idx]}.png')

  

plt.tight_layout()
plt.savefig(f'D:\python\diffusion_model\images\different_scheduler.png')
plt.show()