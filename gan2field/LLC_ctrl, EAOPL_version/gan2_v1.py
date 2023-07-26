import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class Gan2Data(Dataset):
    def __init__(self,path,transform=None):
        self.csv = pd.read_csv(path, index_col=0)
        self.transform = transform
    def __len__(self):
        return len(self.csv)
    def __getitem__(self,idx):
        data = self.csv.iloc[idx].astype("float64")
        data = torch.Tensor(data.values)
        if self.transform:
            data = self.transform(data)
        return data


class Generator(nn.Module):
    def __init__(self, input_dim, z_dim, output_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.Linear(z_dim, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim*2, z_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_dim*4, output_dim)
        )
    def forward(self, noise):
        return self.gen(noise)


class Discriminator(nn.Module):
    def __init__(self,input_dim, z_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.Linear(z_dim, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim*2, z_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_dim*4, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim*2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self,dinput):
        return self.disc(dinput)


def get_noise(sample_size, z_dim):
    return torch.randn(sample_size, z_dim)


def normalizer(data):
    if data[1] == 175.0:
        data[1] = 1.0
    if data[0] == 2.0:
        data[0] = 0.5
    return data


def calculate_reals(batch):
    outputs = torch.empty((0))
    for each in batch:
    	if each[0] != 0.0:
    		tmp = torch.tensor([[1.0]])
    	else:
    		tmp = torch.tensor([[1.0]])
    	outputs = torch.cat((outputs, tmp))
    return outputs


epochs = 15
lr = 0.01
batch_size=128
criterion = nn.BCEWithLogitsLoss()
z_dim = 4
dataset = Gan2Data("./gan2field.csv", normalizer)
dataset = DataLoader(dataset, shuffle = True, batch_size=batch_size, drop_last=True)


gen = Generator(z_dim, 3, 2)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(2, 4)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
disc.load_state_dict(torch.load("./versions/00_model.pt"))
disc_opt.load_state_dict(torch.load("./versions/00_opt.pt"))


mean_disc_losses = []
mean_gen_losses = []
cur_step = 1
display_step = 500

sig = nn.Sigmoid()
for epoch in range(epochs):
    for cur_batch in dataset:
        """
        disc.zero_grad()
        noises = get_noise(batch_size, z_dim)
        fakes = gen(noises)
        fake_predictions = disc(fakes.detach())

        
        fake_1 = torch.empty((100, 2)).uniform_(1,1.5)
        fake_2 = torch.empty((28, 2)).uniform_(-1,0)
        fake_12 = torch.cat((fake_1, fake_2))
        fake_3 = torch.empty((0,2))
        for i in torch.randperm(128):
        	fake_3 = torch.cat((fake_3, fake_12[i].view(-1,2)))
        fake_predictions = disc(fake_3) #torch.empty((0,2))
        
        real_predictions = disc(cur_batch)
        disc_fake_loss = criterion(fake_predictions, torch.zeros_like(fake_predictions))
        disc_real_loss = criterion(real_predictions, calculate_reals(cur_batch))
        disc_loss = (disc_fake_loss + disc_real_loss)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        mean_disc_losses += [disc_loss.item()]
        """
        gen.zero_grad()
        noise = get_noise(batch_size, z_dim)
        fakes = gen(noise)
        fake_predictions = disc(fakes)
        gen_loss = criterion(fake_predictions, torch.ones_like(fake_predictions))
        gen_loss.backward()
        gen_opt.step()

        mean_gen_losses += [gen_loss.item()]
        
        if cur_step%display_step==0:
            mean_gen_loss = sum(mean_gen_losses[-display_step:]) / display_step
            mean_disc_loss = sum(mean_disc_losses[-display_step:]) / display_step
            out = "Epoch {} : gen loss = {} -- disc loss = {}".format(epoch, mean_gen_loss, mean_disc_loss)
            print(out)
            noise = get_noise(1, z_dim)
            print("Sample : ", end="")
            fake = gen(noise)
            print(fake)
            print("Disc : ", end="")
            print(sig(disc(fake)).item())

            """
            print("EAPOL : ", end="")
            epoal = float(input())
            print("LLC : ", end="")
            llc = float(input())
            print("Disc out : ", end="")
            print(sig(disc(torch.tensor([epoal, llc]))).item())
			"""
            print("\nEPOAL  |  LLC")
            for i in [x/10 for x in range(-10,20)]:
                inp = "  " + str(i) + "  "
                print(inp+"|   0      ", end="")
                t = torch.tensor([float(i), float(0)])
                print(sig(disc(t)).item())
            fake_1 = torch.empty((1, 2)).uniform_(1,2)
            fake_1[0][1] = 0.0
            print("{:.2f}".format(fake_1[0][0].item()), end = "  |  ")
            print("0", end= "    ")
            print(sig(disc(fake_1)).item(), end="\n\n")
        if cur_step%100 == 0:
            print(cur_step)
        cur_step += 1

inp_range = torch.arange(-1,2,0.1)
disc_inp = torch.empty((0,2))
x = torch.empty((0))
y = torch.empty((0))
for i in inp_range:
    for j in inp_range:
        epoal = torch.tensor([[i.item()]])
        llc = torch.tensor([[j.item()]])
        disc_inp = torch.cat((disc_inp, torch.cat((epoal, llc), dim=1)))
        x = torch.cat((x, torch.tensor([i.item()])))
        y = torch.cat((y, torch.tensor([j.item()])))
col = sig(disc(disc_inp)) * 100
plt.scatter(x, y, c=col.detach().numpy(), cmap="autumn", marker="*", s=10)
plt.title("Disc predictions")
plt.xlabel("EAPOL")
plt.ylabel("LLC")
plt.colorbar()
plt.show()

save = input("Save(Y/N): ")
if save.lower() == "y":
	torch.save(disc.state_dict(), "./versions/01_disc_model.pt")
	torch.save(disc_opt.state_dict(), "./versions/01_disc_opt.pt")
	torch.save(gen.state_dict(), "./versions/01_gen_model.pt")
	torch.save(gen_opt.state_dict(), "./versions/01_gen_opt.pt")