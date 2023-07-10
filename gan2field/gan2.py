import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from viewer import view2d


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
            nn.Tanh(),
            nn.Linear(z_dim*2, z_dim*4),
            nn.Tanh(),
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
            nn.Tanh(),
            nn.Linear(z_dim*2, z_dim*2),
            nn.Tanh(),
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

disc.load_state_dict(torch.load("./versions/03_disc_model.pt"))
disc_opt.load_state_dict(torch.load("./versions/03_disc_opt.pt"))

mean_disc_losses = []
mean_gen_losses = []
cur_step = 1
display_step = 500

print(disc)
p = int(input("Num:"))
print(disc.disc[p].weight)

sig = nn.Sigmoid()

view2d(disc, "EAPOL_version", "LLC_ctrl", -0.5,1.5,0.05)

for epoch in range(epochs):
    for cur_batch in dataset:
        disc.zero_grad()
        noises = get_noise(batch_size, z_dim)
        fakes = gen(noises)
        fake_predictions = disc(fakes.detach())
        real_predictions = disc(cur_batch)
        fake_disc_loss = criterion(fake_predictions, torch.zeros_like(fake_predictions))
        real_disc_loss = criterion(real_predictions, calculate_reals(cur_batch))
        disc_loss = (fake_disc_loss + real_disc_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        mean_disc_losses += [disc_loss.item()]
        
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
            fake = gen(noise)
            print("Sample : ", end="")
            print(fake)
            print("Disc out : ", end="")
            print(sig(disc(fake)).item())
        if cur_step%100 == 0:
            print(cur_step)
        if cur_step%2000==0:
            view2d(disc, -1,2.5,0.07)
            save = input("Save(Y/N): ")
            if save.lower() == "y":
                torch.save(disc.state_dict(), "./versions/03_disc_model.pt")
                torch.save(disc_opt.state_dict(), "./versions/03_disc_opt.pt")
                torch.save(gen.state_dict(), "./versions/03_gen_model.pt")
                torch.save(gen_opt.state_dict(), "./versions/03_gen_opt.pt")
        cur_step += 1

