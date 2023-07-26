import matplotlib.pyplot as plt
import torch
from torch import nn

sig = nn.Sigmoid()
def view2d(disc, xlabel, ylabel, a, b, step=1.0):
	inp_range = torch.arange(a,b,step)
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
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.colorbar()
	plt.show()
    
