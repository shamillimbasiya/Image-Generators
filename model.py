import torch

nn = torch.nn
F = nn.functional

class QuadConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels = None, residual = False):
        super().__init__()
        self.out_channels = out_channels
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        self.quadconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(1,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
        )
        
    def forward(self, x):
        out = self.quadconv(x)
        out = F.layer_norm(out,[self.out_channels,out.size(-2),x.size(-1)])
        return F.mish(out) if not self.residual else F.gelu(x + out)

class Encoder(nn.Module):

    def __init__(self, channel_list, sample_dim, embed_dim):
        super().__init__()
        
        #self.quadconv1 = QuadConv(channel_list[0], channel_list[1])
        #self.pool = nn.MaxPool2d(2)
        #self.quadconv2 = QuadConv(channel_list[1], channel_list[2])
    
        #self.quadconv3 = QuadConv(channel_list[2],channel_list[3])
        #self.quadconv4 = QuadConv(channel_list[3],channel_list[4])
        #self.quadconv5 = QuadConv(channel_list[4],16)
        self.down1 = Down(channel_list[0],channel_list[1], embed_dim)
        self.down2 = Down(channel_list[1],channel_list[2], embed_dim)
        #self.down3 = Down(channel_list[2],channel_list[3], embed_dim)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(4096, sample_dim),
        )
        

    def forward(self, x, y):
        
        #x1 = self.quadconv1(x)
        #x2 = self.pool(self.quadconv2(x1))
        #x3 = self.pool(self.quadconv3(x2))
        #x4 = self.pool(self.quadconv4(x3))
        #x5 = self.pool(self.quadconv5(x4))
        x = self.down1(x,y)
        x = self.down2(x,y)
        #x = self.down3(x,y)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x #self.encode(x)



class Decoder(nn.Module):
    def __init__(self, channel_list, latent_dim, embed_dim):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 4096),
        )
        self.unflatten =  nn.Unflatten(-1,(64,8,8))
        #self.upsample =  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #self.quadconv1 =  QuadConv(64, channel_list[4])
        #self.quadconv2 =  QuadConv(channel_list[4], channel_list[3])
        #self.quadconv3 =  QuadConv(channel_list[3], channel_list[2])
        #self.quadconv4 =  QuadConv(channel_list[2], channel_list[1])
        #self.quadconv5 =  QuadConv(channel_list[1], channel_list[0])
       
        #self.upsample1 = Up(channel_list[4],channel_list[3],nr_classes, embed_dim)
        #self.upsample1 = Up(channel_list[3],channel_list[2], embed_dim)
        self.upsample2 = Up(channel_list[2],channel_list[1], embed_dim)
        self.upsample3 = Up(channel_list[1],channel_list[0], embed_dim)
        
        
        
    def forward(self, z, y):
        
        z = self.linear(z)
        z = self.unflatten(z)
        #z = self.upsample1(z, y)
        z = self.upsample2(z, y)
        z = self.upsample3(z, y)

        return z #self.decode(z)


class Sampler(nn.Module):

    def __init__(self, in_dim, latent_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.logvar = nn.Linear(in_dim, latent_dim)
        self.mean = nn.Linear(in_dim, latent_dim)
        self.embed = nn.Linear(embed_dim, latent_dim)

    def forward(self, x, y):
        
        return self.logvar(x), self.mean(x), self.embed(F.one_hot(y,self.embed_dim).float())
    
class Up(nn.Module):

    def __init__(self, inc, outc, embedding_dim):
        super().__init__()
        
        self.upsample = nn.Sequential(
            QuadConv(inc, outc),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x, y):       
        x = self.upsample(x)
        #y = self.embed(y)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        
        return x

class Down(nn.Module):

    def __init__(self, inc, outc, embedding_dim):
        super().__init__()
        
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            QuadConv(inc, outc)
        )

    def forward(self, x, y):
        x = self.downsample(x)
        #y = self.embed(y)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        
        return x
    
class VariationalAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        channel_list = [3,128,64,512]
        self.latent_dim = 256#1024
        self.sample_dim = 64#256
        self.classes = 10
        self.embedding_dim = 11
        #self.embed = nn.Embedding(self.classes, self.embedding_dim)
        #self.one = F.one_hot()
        self.encoder = Encoder(channel_list, self.sample_dim, self.embedding_dim)
        self.sampler = Sampler(self.sample_dim, self.latent_dim, self.embedding_dim)
        self.decoder = Decoder(channel_list, self.latent_dim, self.embedding_dim)
        #self.classEncoder = ClassEncoder(self.classes, self.embedding_dim, self.sample_dim)

    def calculateSample(self, logvar, mean, y):
        epsilon = torch.randn_like(logvar)

        return mean + torch.exp(0.5 * logvar)*epsilon + 20*y
    
    def forward(self, x, y):
        #y = self.embed(y)
        encoded_image = self.encoder(x, y)
        logvar, mean, y = self.sampler(encoded_image, y)
        sample = self.calculateSample(logvar, mean, y)
        decoded = self.decoder(sample, y)

        return F.sigmoid(decoded), logvar, mean
    
    def generate(self, device, y):
        #y = self.embed(y)
        distribution = torch.randn(1,self.sample_dim).to(device)
        logvar, mean, y = self.sampler(distribution, y)
        sample = self.calculateSample(logvar, mean, y)
        decoded = self.decoder(sample, y)

        return F.sigmoid(decoded)
