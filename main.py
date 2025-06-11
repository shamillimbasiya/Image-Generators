from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
nn = torch.nn
F = nn.functional
import math
from model import VariationalAutoencoder
from torch.utils.data import Subset

# --- Specify hyper-parameters ---
batch_size = 46
num_classes = 10
num_epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
trainset, testset, classes = load_CIFAR10(num_classes=10)

trainset = getSubset(trainset, 100)

print("Classes:", classes)
print("Number of training images:", len(trainset))
print("Number of test images:", len(testset))

trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)



def train(epochs, lr, train_data, model):
    
    torch.cuda.empty_cache()
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print('Epoch: ', epoch+1)
        pbar = tqdm(train_data)
        average_loss = 0
        for e,(x,y) in enumerate(pbar,1):
            x = x.to(device)
            y = y.to(device)
            #print(y)
            pred, logvar, mean = model(x, y)
            BCE = F.binary_cross_entropy(pred, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            
            loss = BCE*0.8 + KLD *0.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()/batch_size
            average_loss += loss_item
            BCE_item = BCE.item()/batch_size
            KLD_item = KLD.item()/batch_size
            pbar.set_postfix({
                              'Loss': loss_item, 
                              'BCE_Loss': BCE_item, 
                              'KLD_Loss': KLD_item
                              })
        print(average_loss/(e*batch_size))

     

    return model

#model = train(200, 1e-3, trainloader, VariationalAutoencoder())
#torch.save(model.state_dict(), 'VAE_v2_1.pt')

def compute_ssim(original_images, model):
    ssim_values = []
    for e,(x,y) in enumerate(original_images):
        #print(x.shape)
        original_image = x.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        
        # Generate image
        generated_image = model.generate(device).cpu().squeeze(0).permute(1, 2, 0).numpy() 
      
        #print(original_image.shape)
        #print(generated_image.shape)
        
        # Ensure both images have the same data type
        original_image = original_image.astype(np.uint8)
        generated_image = generated_image.astype(np.uint8)
        

        ssim_value = ssim(original_image, generated_image, multichannel=True, channel_axis=-1)
        ssim_values.append(ssim_value) 
    
    return ssim_values

model = VariationalAutoencoder().to(device)
model.load_state_dict(torch.load('Initial_VAE_model.pt'))
images = []
ssim_values = []
test_images = [(x, y) for (x, y) in testset]
with torch.inference_mode():
    ssim_values = compute_ssim(test_images, model)
print("Average SSIM:", np.mean(ssim_values))

with torch.inference_mode():
    obj = torch.ones(1,dtype=int).to(device)*0
    rows = 10
    for i in range(rows**2):
        if i%rows == 0:
            obj += 1
            #print(obj)
        generated_image = model.generate(device, obj)
        images.append(generated_image.cpu())
show_images(images)
