# ----------------------------------------------------------------------------------
# Imports
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as torchFunc
import torch.autograd as autograd
import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import pickle
from collections import namedtuple
import argparse

# ----------------------------------------------------------------------------------
# Global parameters
parser = argparse.ArgumentParser(description='HW4 EX.4 - GAN Performances')
parser.add_argument('--mode', type=str, default='train',
                    help='running mode: (train, eval)')
parser.add_argument('--model', type=str, default='DCGAN',
                    help='type of GAN (DCGAN, WGAN-GP)')
parser.add_argument('--use_batchnorm', type=str, default='true',
                    help='use BatchNorm layers? (true, false)')
parser.add_argument('--batch_size', type=float, default=50,
                    help='batch size')
parser.add_argument('--num_epochs', type=float, default=10,
                    help='number of epochs to train')
parser.add_argument('--gp_lambda', type=float, default=10,
                    help='Gradient Penalty parameter')
parser.add_argument('--disc_iters', type=float, default=5,
                    help='discriminator runs for each generator run')
args = parser.parse_args()

env = namedtuple('env',[])
env.mode = args.mode
env.model = args.model
env.use_batchnorm = args.use_batchnorm
env.batch_size = args.batch_size
env.num_epochs = args.num_epochs
env.gp_lambda = args.gp_lambda
env.disc_iters = args.disc_iters
env.saved_workspace = r"\workspace.bin"
#--------------------------------------
env.data = 'data'
env.input_size = 200
env.hidden_layers_num = 200
env.layers_num = 2
env.epochs = 20
env.seed = 123
env.log_interval = 200
env.resume = ''
env.optimizer = 'sgd'
env.cuda = torch.cuda.is_available()
#--------------------------------------

# ----------------------------------------------------------------------------------
# Helper functions and stuff
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([My_gen, My_disc, criterion, optimizer], f)

def model_load(fn):
    global My_gen, My_disc, criterion, optimizer
    with open(fn, 'rb') as f:
        My_gen, My_disc, criterion, optimizer = torch.load(f)

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if env.cuda:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

torch.manual_seed(env.seed)
def generate_noise(length):
    return torch.normal(torch.zeros(env.batch_size,length))


# ----------------------------------------------------------------------------------
# Data loading
train_dataset = dsets.FashionMNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
test_dataset = dsets.FashionMNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=env.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=env.batch_size,
                                          shuffle=False)

num_images = train_dataset.train_data.shape[0]
image_x = train_dataset.train_data.shape[1]
image_y = train_dataset.train_data.shape[2]


# ----------------------------------------------------------------------------------
# Building the model
def init_weights_norm(some_layer):
    init_range = 0.02
    if (type(some_layer) in [nn.Linear, nn.BatchNorm2d, nn.ConvTranspose2d, nn.Conv2d]):
        some_layer.bias.data.fill_(0)
        some_layer.weight.data.normal_(0, init_range)

class GAN_Generator(nn.Module):
    def __init__(self, input_size, use_batchnorm):
        super(GAN_Generator, self).__init__()
        self.L1 = nn.Linear(input_size, 4*4*(4*64))
        self.TConv1 = nn.ConvTranspose2d(4*64, 2*64, [5, 5], stride=2, padding=2)
        self.BN1 = nn.BatchNorm2d(2*64)
        self.TConv2 = nn.ConvTranspose2d(2*64, 64, [5, 5], stride=2, padding=1)
        self.BN2 = nn.BatchNorm2d(64)
        self.TConv3= nn.ConvTranspose2d(64, 1, [5, 5], stride=2, padding=1)

        self.apply(init_weights_norm)
        self.use_batchnorm = use_batchnorm

    def forward(self, input):
        x = torchFunc.relu(self.L1(input))
        x = x.view(x.size(0),4*64,4,4)

        x = self.TConv1(x)
        if self.use_batchnorm:
            x = self.BN1(x)
        x = torchFunc.relu(x)

        x = self.TConv2(x)
        x = (x.narrow(2,0,14)).narrow(3,0,14)
        if self.use_batchnorm:
            x = self.BN2(x)
        x = torchFunc.relu(x)

        x = torchFunc.relu(self.TConv3(x))
        x = (x.narrow(2, 0, 28)).narrow(3, 0, 28).contiguous()
        output = torch.sigmoid(x.view(x.size(0),784))
        return output


class GAN_Discriminator(nn.Module):
    def __init__(self, use_batchnorm):
        super(GAN_Discriminator, self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, [5, 5], stride=2, padding=2)
        self.Conv2 = nn.Conv2d(64, 128, [5, 5], stride=2, padding=2)
        self.BN1 = nn.BatchNorm2d(128)
        self.Conv3 = nn.Conv2d(128, 256, [5, 5], stride=2, padding=2)
        self.BN2 = nn.BatchNorm2d(256)
        self.L1 = nn.Linear(4*4*(4*64), 1)

        self.apply(init_weights_norm)
        self.use_batchnorm = use_batchnorm

    def forward(self, input):
        x = input.view(input.size(0), 1, 28, 28)

        x = torchFunc.leaky_relu(self.Conv1(x))

        x = self.Conv2(x)
        if self.use_batchnorm:
            x = self.BN1(x)
        x = torchFunc.leaky_relu(x)

        x = self.Conv3(x)
        if self.use_batchnorm:
            x = self.BN2(x)
        x = torchFunc.leaky_relu(x)

        x = x.view(x.size(0), -1)
        output = self.L1(x)

        return output


My_gen = GAN_Generator(128,env.use_batchnorm)
My_disc = GAN_Discriminator(env.use_batchnorm)
criterion = nn.CrossEntropyLoss()
if env.model == 'WGAN-GP':
    #TODO: maybe make them the same..
    gen_optimizer = torch.optim.Adam(My_gen.parameters(),lr=1e-4, betas=[0.5, 0.9])
    disc_optimizer = torch.optim.Adam(My_disc.parameters(),lr=1e-4, betas=[0.5, 0.9])
elif env.model == 'DCGAN':
    gen_optimizer = torch.optim.Adam(My_gen.parameters(), lr=2e-4, betas=[0.5, 0.999])
    disc_optimizer = torch.optim.Adam(My_gen.parameters(), lr=2e-4, betas=[0.5, 0.999])
else:
    assert False, "Wrong model"

if env.cuda:
    My_gen = My_gen.cuda()
    My_disc = My_disc.cuda()
    criterion = criterion.cuda()

###
#TODO
if env.resume:
    print('Resuming model ...')
    model_load(env.saved_workspace)
###

###
gen_params = list(My_gen.parameters())
total_gen_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in gen_params if x.size())
print('Generator model total parameters:', total_gen_params)
disc_params = list(My_disc.parameters())
total_disc_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in disc_params if x.size())
print('Discriminator model total parameters:', total_disc_params)

# ----------------------------------------------------------------------------------
# Loss calculation
def calc_gp(real_data, fake_data):
    #call disc_optimizer.zero_grad() before me!
    #call disc_optimizer.zero_grad() after me!
    alpha = torch.rand(env.batch_size)
    differences = fake_data - real_data
    a=torch.t(differences)
    b=torch.diag(alpha)
    interpolates = real_data + torch.t(torch.matmul(a,b))
    interpolates = Variable(interpolates, requires_grad=True)

    output = My_disc(interpolates)
    output.backward(torch.ones(interpolates.size(0),1))
    gradients = interpolates.grad

    slopes = torch.sqrt(torch.sum(torch.pow(gradients,2),dim=1))
    return torch.mean((slopes - 1.) ** 2)

def calc_gen_loss(gan_disc_outputs):
    if env.model == 'WGAN-GP':
        gen_loss  = -gan_disc_outputs
    elif env.model == 'DCGAN':
        fake_outputs = torch.sigmoid(gan_disc_outputs.view(-1))
        out_fake_vec = torch.zeros(env.batch_size,2)
        out_fake_vec[:,0] = fake_outputs
        out_fake_vec[:, 1] = 1 - fake_outputs

        one_vec = torch.ones([env.batch_size],dtype=torch.long)

        gen_loss = criterion(out_fake_vec,one_vec)
    else:
        assert False, "Wrong model"
    return gen_loss

def calc_disc_loss(real_outputs,fake_outputs, gradient_penalty=0):
    if env.model == 'WGAN-GP':
        disc_loss = fake_outputs - real_outputs + env.gp_lambda*gradient_penalty
    elif env.model == 'DCGAN':
        fake_outputs = torch.sigmoid(fake_outputs.view(-1))
        real_outputs = torch.sigmoid(real_outputs.view(-1))
        out_fake_vec = torch.zeros(env.batch_size,2)
        out_real_vec = torch.zeros(env.batch_size, 2)
        out_fake_vec[:,0] = fake_outputs
        out_fake_vec[:, 1] = 1 - fake_outputs
        out_real_vec[:, 0] = real_outputs
        out_real_vec[:, 1] = 1 - real_outputs

        one_vec = torch.ones([env.batch_size],dtype=torch.long)

        disc_loss = criterion(out_fake_vec,1-one_vec) + \
                    criterion(out_real_vec,one_vec)
    else:
        assert False, "Wrong model"
    return disc_loss

###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    print("hi")



def train(cur_epoch):
    for batch, (images, labels) in enumerate(train_loader):
        disc_optimizer.zero_grad()

        noise_batch = generate_noise(128)
        with torch.no_grad():
            fake_images = My_gen(noise_batch)
        real_images = images.view(images.size(0),-1)

        gp = calc_gp(real_images, fake_images)
        disc_optimizer.zero_grad()

        out_real = My_disc(real_images)
        out_fake = My_disc(fake_images)
        disc_loss = calc_disc_loss(out_real, out_fake, gp)
        disc_loss.backward()
        disc_optimizer.step()
        if (batch % env.disc_iters == 0):
            gen_optimizer.zero_grad()

            noise_batch = generate_noise(128)
            fake_images = My_gen(noise_batch)
            # with torch.no_grad():
            out_fake = My_disc(fake_images)
            gen_loss = calc_gen_loss(out_fake)
            gen_loss.backward()
        if (batch*env.batch_size % (num_images/60) < env.batch_size):
            print("[Epoch=" + str(cur_epoch) + ",Batch=" + str(batch) + \
                  ",   disc_loss={:6.3f}, gen_loss={:6.3f}]".format(disc_loss.data.item(), gen_loss.data.item()))
# ----------------------------------------------------------------------------------
# GO

if (env.mode == "eval"):
    print("hi")
# TODO
# my_net.load_state_dict(torch.load('model.pkl'))
else:
    for epoch in range(env.epochs):
        train(epoch)
        print('-' * 89)
        print('| End of epoch {:3d} |'.format(epoch))
        print('-' * 89)

#save the model

# ----------------------------------------------------------------------------------
# eval

#load the model
#eval the test data

# ----------------------------------------------------------------------------------
# ending
#make a graph