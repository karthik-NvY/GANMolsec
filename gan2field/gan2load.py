import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from viewer import view2d

class Discriminator(nn.Module):
    def __init__(self,input_dim, z_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(input_dim, z_dim),
            nn.Linear(z_dim, z_dim*2),
            nn.Tanh(),
            nn.Linear(z_dim*2, z_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_dim*4, z_dim*2),
            nn.Tanh(),
            nn.Linear(z_dim*2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self,dinput):
        return self.disc(dinput)


sig = nn.Sigmoid()
disc = Discriminator(2, 4)
disc_opt = torch.optim.Adam(disc.parameters(), lr=1)


batch_size = 128
criterion = nn.BCEWithLogitsLoss()
cur_step = 1
display_step = 30
mean_disc_losses = []
for i in range(10000):
	disc.zero_grad()
	fake_1 = torch.empty((100, 2)).uniform_(1,1.5)
	fake_2 = torch.empty((28, 2)).uniform_(-1,0)
	fake_12 = torch.cat((fake_1, fake_2))
	fake_3 = torch.empty((0,2))
	for i in torch.randperm(128):
		fake_3 = torch.cat((fake_3, fake_12[i].view(-1,2)))
	fake_predictions = disc(fake_3)

	reals0 = torch.full((64,1), 0.0)
	reals1 = torch.full((64,1), 1.0)
	reals = torch.cat((reals0, reals1))
	llc = torch.zeros_like(reals)
	reals = torch.cat((reals, llc), dim=1)
	real_predictions = disc(reals)
	
	fake_disc_loss = criterion(fake_predictions, torch.zeros_like(fake_predictions))
	real_disc_loss = criterion(real_predictions, torch.ones_like(real_predictions))
	disc_loss = (fake_disc_loss + real_disc_loss) / 2
	disc_loss.backward()
	disc_opt.step()

	mean_disc_losses += [disc_loss.item()]
	if cur_step%display_step == 0:
		mean_loss = sum(mean_disc_losses[-display_step:]) / display_step
		print("Epoch : {} -- disc_loss : {} ".format(i, mean_loss))
		for i in [0,1]:
			for j in [0,0.5,1]:
				inp = "  " +str(j) + "  "
				if j!=0.5:
					inp +="  "
				print(inp+"|   "+str(i)+"      ", end="")
				t = torch.tensor([float(j), float(i)])
				print(sig(disc(t)).item())
		print("")
		for i in [x/10 for x in range(-20,20)]:
			inp = "  " + str(i) + "  "
			print(inp+"|   0      ", end="")
			t = torch.tensor([float(i), float(0)])
			print(sig(disc(t)).item())
		fake_1 = torch.empty((1, 2)).uniform_(1,2)
		fake_1[0][1] = 0.0
		print("{:.2f}".format(fake_1[0][0].item()), end = "  |  ")
		print("0", end= "    ")
		print(sig(disc(fake_1)).item(), end="\n\n")
	if (cur_step % 5)==0:
    		print(cur_step)
	cur_step+=1

view2d(disc, -7, 7, 0.2)

inp = input("Save(Y/N)? : ")
if inp.lower() == "y":
	torch.save(disc.state_dict(), "./versions/02_disc_model.pt")
	torch.save(disc_opt.state_dict(), "./versions/02_disc_opt.pt")

while(0):
	epoal = float(input("EAPOL : "))
	llc = float(input("LLC : "))
	print("Disc out : ", end="")
	print(sig(disc(torch.tensor([epoal, llc]))).item())