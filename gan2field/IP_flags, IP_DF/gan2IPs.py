import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from helper import view2d

# Features:2
"""
Num : 8 ----  IP_flags
3    76110 0
2     5694 0.5
1     2090
Name: IP_flags, dtype: int64
Max : 3
Min : 1
Mean : 2.882303859632393

Num : 9 ----  IP_DF
1    76110
0     7784
Name: IP_DF, dtype: int64
Max : 1
Min : 0
Mean : 0.9072162490762152
"""


class Gan2Data(Dataset):
    def __init__(self,path,transform=None):
        """
            path(string): path to a csv file.
            transform(callable): a function to apply some transform on data.
        """
        self.csv = pd.read_csv(path, index_col=0)
        self.transform = transform
    def __len__(self):
        """
            returns the length of dataset.
        """
        return len(self.csv)
    def __getitem__(self,idx):
        """
            idx(int): index in the pandas.dataframe

            returns a datapoint after applying transformation(if available).
        """
        data = self.csv.iloc[idx].astype("float64")
        data = torch.Tensor(data.values)
        if self.transform:
            data = self.transform(data)
        return data

# Generator Class
class Generator(nn.Module):
    def __init__(self, input_dim, z_dim, output_dim):
        """
            gen_parameters(list): list object with following contraints
                             gen_parameters[0] will be the input dimension(latent)
                             gen_parameters[-1] will be the output dimension
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.Linear(z_dim, z_dim*2),
            nn.Tanh(),
            nn.Linear(z_dim*2, z_dim*4),
            nn.Tanh(),
            nn.Linear(z_dim*4, z_dim*2),
            nn.Tanh(),
            nn.Linear(z_dim*2, output_dim)
        )
    def forward(self, noise):
        """
            noise(tensor) : input to NN.

            returns output after running noise through self.gen
        """
        return self.gen(noise)

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self,input_dim, z_dim):
        """
            disc_parameters(list): list object with following contraints
                             disc_parameters[0] will be the input dimension(latent)
                             disc_parameters[-1] will be the output dimension
        """
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
        """
            dinput(tensor) : input to disc NN.

            returns output after running noise through self.disc
        """
        return self.disc(dinput)


def get_noise(sample_size, z_dim):
    """
        sample_size(int): dimension to return noise
        z_dim(int) : dimension to return noise

        returns a random tensor of shape (sample_size, z_dim).
    """
    return torch.randn(sample_size, z_dim)


def normalizer(data):
    """
        This is the transformation function for Gan2Data class.
        data(tensor) : data to be transformed.

        returns data after transformation.
    """
    if data[0] == 3.0:
        data[0] = 0.0
    elif data[0] == 2.0:
        data[0] = 0.5
    return data


def calculate_reals(batch):
    """
    returns ground truth labels for batch.
    """
    outputs = torch.empty((0))
    for each in batch:
        if each[0] != 0.0:
            tmp = torch.tensor([[1.0]])
        else:
            tmp = torch.tensor([[1.0]])
        outputs = torch.cat((outputs, tmp))
    return outputs


epochs = 16
lr = 0.01
batch_size=128
criterion = nn.BCEWithLogitsLoss()
z_dim = 4 # Latent Dimension

# Dataset initialization
dataset = Gan2Data("./IPs.csv", normalizer)
dataset = DataLoader(dataset, shuffle = True, batch_size=batch_size, drop_last=True)

# Generator
gen = Generator(z_dim, 3, 2)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

# Discriminator
disc = Discriminator(2, 4)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

#disc.load_state_dict(torch.load("./versions/04_disc_model.pt"))
#disc_opt.load_state_dict(torch.load("./versions/04_disc_opt.pt"))
#gen.load_state_dict(torch.load("./versions/041_gen_model.pt"))
#gen_opt.load_state_dict(torch.load("./versions/041_gen_opt.pt"))

mean_disc_losses = [] # keepts track of generator losses.
mean_gen_losses = [] # keepts track of discriminator losses.
cur_step = 1
display_step = 500

"""
print(disc_opt.state_dict()["param_groups"])

print(disc)
p = int(input("Num:"))
print(disc.disc[p].weight)
"""

sig = nn.Sigmoid()
view2d(disc, "IP_flags", "IP_DF", -1,4,0.1)

for epoch in range(epochs):
    for cur_batch in dataset:

        # Discriminator Training
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
        
        # Generator Training
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
            view2d(disc, "IP_flags", "IP_DF", -1,4,0.1)
            save = input("Save(Y/N): ")
            if save.lower() == "y":
                torch.save(disc.state_dict(), "./versions/041_disc_model.pt")
                torch.save(disc_opt.state_dict(), "./versions/041_disc_opt.pt")
                torch.save(gen.state_dict(), "./versions/041_gen_model.pt")
                torch.save(gen_opt.state_dict(), "./versions/041_gen_opt.pt")
        cur_step += 1
view2d(disc, "IP_flags", "IP_DF", -1,4,0.1)
save = input("Save(Y/N): ")
if save.lower() == "y":
    torch.save(disc.state_dict(), "./versions/041_disc_model.pt")
    torch.save(disc_opt.state_dict(), "./versions/041_disc_opt.pt")
    torch.save(gen.state_dict(), "./versions/041_gen_model.pt")
    torch.save(gen_opt.state_dict(), "./versions/041_gen_opt.pt")
