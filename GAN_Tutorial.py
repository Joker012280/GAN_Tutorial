import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# is_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if is_cuda else 'cpu')
device = torch.device('cpu')

## Standardization
# standard = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
standard = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5),std=(0.5))])


## Mnist Data

train_data = dsets.MNIST(root ='data/', train=True, transform=standard, download=True)
test_data = dsets.MNIST(root='data/', train=False, transform=standard, download=True)

batch_size = 150
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle = True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle = True)

def imshow(img) :
    img = (img+1)/2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = utils.make_grid(img.cpu().detach())
    img = (img+1)/2
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1,2,0)))
    plt.show()

# Config
noise = 100
hidden = 128

def sample_z(batch_size = 1 , noise = noise) :
    return torch.randn(batch_size,noise,device = device)

Generator = nn.Sequential(
    nn.Linear(noise,hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden,hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden,28*28),
    nn.Tanh()
)

Discriminator = nn.Sequential(
    nn.Linear(28*28,hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden,hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()

def run_epoch(gen,discrim,g_optim,d_optim) :
    gen.train()
    discrim.train()


    for img_batch, label_batch in train_data_loader:

        d_optim.zero_grad()

        prob_real = discrim(img_batch.view(-1,28*28))
        prob_fake = discrim(gen(sample_z(batch_size,noise)))

        loss_d = criterion(prob_real,torch.ones_like(prob_real)) + \
            criterion(prob_fake,torch.zeros_like(prob_fake))

        loss_d.backward()
        d_optim.step()


        g_optim.zero_grad()

        prob_fake = discrim(gen(sample_z(batch_size,noise)))

        loss_g = criterion(prob_fake,torch.ones_like(prob_fake))

        loss_g.backward()
        g_optim.step()


def evaluate_model(generator, discriminator):
    p_real, p_fake = 0., 0.

    generator.eval()
    discriminator.eval()

    for img_batch, label_batch in test_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28 * 28))).item()) / 10000.
            p_fake += (torch.sum(discriminator(generator(sample_z(batch_size, noise)))).item()) / 10000.

    return p_real, p_fake


def init_params(model):
    for p in model.parameters():
        if (p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)


init_params(Generator)
init_params(Discriminator)

optimizer_g = optim.Adam(Generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(Discriminator.parameters(), lr=0.0002)

p_real_trace = []
p_fake_trace = []

for epoch in range(200):
    run_epoch(Generator, Discriminator, optimizer_g, optimizer_d)
    p_real, p_fake = evaluate_model(Generator, Discriminator)

    p_real_trace.append(p_real)
    p_fake_trace.append(p_fake)

    if ((epoch + 1) % 25 == 0):
        print('(epoch %i/200) p_real: %f, p_g: %f' % (epoch + 1, p_real, p_fake))
        imshow_grid(Generator(sample_z(16)).view(-1, 1, 28, 28))


plt.plot(p_fake_trace, label='D(x_generated)')
plt.plot(p_real_trace, label='D(x_real)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


vis_loader = torch.utils.data.DataLoader(test_data, 16, True)
img_vis, label_vis   = next(iter(vis_loader))
imshow_grid(img_vis)

imshow_grid(Generator(sample_z(16,100)).view(-1, 1, 28, 28))