import matplotlib.pyplot as plt
import torch
import os
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
	plt.scatter(x, y, c=col.detach().numpy(), cmap="autumn", marker="x", s=10)
	plt.title("Disc predictions")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.colorbar()
	plt.show()


def ip2save(generators, disc, features, path, ver):
	for i, mygen in enumerate(generators):
		folder = os.path.join(path, "gen_" + str(i))
		model = os.path.join(folder, str(ver)+"_gen_model.pt")
		opt = os.path.join(folder, str(ver)+"_gen_opt.pt")
		info = os.path.join(folder, "info.txt")
		if not os.path.exists(folder):
			os.mkdir(folder)
		torch.save(mygen.gen.state_dict(), model)
		torch.save(mygen.gen_opt.state_dict(), opt)
		with open(info, "w") as f:
			change = ""
			for k in range(len(mygen.direction)):
				change+=features[mygen.direction[k]]
				change+="{} ".format(mygen.distance[k])
			f.write("Shift: " + change + "\n")
			noise = torch.randn(1, mygen.gen.z_dim)
			fake = mygen.gen(noise)
			mygen.shift_gen_samples(fake)
			f.write("gen {} : {}\n".format(i, fake))
			f.write("Disc out : ")
			f.write(str(sig(disc[0](fake)).item()))
	disc_path = os.path.join(path, "disc")
	disc_model = os.path.join(disc_path, str(ver) + "_disc_model.pt")
	disc_opt = os.path.join(disc_path, str(ver) + "_disc_opt.pt")
	if not os.path.exists(disc_path):
		os.mkdir(disc_path)
	torch.save(disc[0].state_dict(), disc_model)
	torch.save(disc[1].state_dict(), disc_opt)
