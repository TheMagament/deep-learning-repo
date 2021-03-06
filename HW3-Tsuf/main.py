# ----------------------------------------------------------------------------------
# Imports
import time
import math
import torch
import torch.nn as nn
import os
import DataHandler
import model
import pickle
from collections import namedtuple
from utils import batchify, get_batch, repackage_hidden
import argparse


# ----------------------------------------------------------------------------------
# Global paramerets
parser = argparse.ArgumentParser(description='OUR MaGNIFIceNT NEtS!!')
parser.add_argument('--mode', type=str, default='eval',
                    help='running mode: train | eval | generate')
parser.add_argument('--method', type=str, default='WGAN-GP',
                    help='type of network (WGAN-GP, DCGAN)')
parser.add_argument('--lr', type=float, default=30,
                    help='learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient penalty')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay')
args = parser.parse_args()

env = namedtuple('env',[])
env.mode = args.mode
env.model = args.model
env.dropout = args.dropout
env.lr = args.lr
env.clip = args.clip
env.wdecay = args.wdecay

env.data = 'data'
env.input_size = 200
env.hidden_layers_num = 200
env.layers_num = 2
env.epochs = 20
env.batch_size = 20
env.seq_len = 35
env.seed = 123
env.log_interval = 200
env.resume = ''
env.optimizer = 'sgd'


env.cuda = torch.cuda.is_available()
env.dropouth = env.dropout
env.dropouti = env.dropout
# ----------------------------------------------------------------------------------
# Data loading

# Functions for saving and loading the model
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([my_model, criterion, optimizer], f)

def model_load(fn):
    global my_model, criterion, optimizer
    with open(fn, 'rb') as f:
        my_model, criterion, optimizer = torch.load(f)

# Convert the data text files into something that is easy to work with
f_name = 'encoded_data'
if os.path.exists(f_name):
    print('Loading cached dataset...')
    allData = torch.load(f_name)
    print(allData.train)
else:
    print('Producing dataset...')
    allData = DataHandler.Corpus(env.data)
    torch.save(allData, f_name)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(allData.train, env.batch_size, env)
val_data = batchify(allData.valid, eval_batch_size, env)
test_data = batchify(allData.test, test_batch_size, env)

# ----------------------------------------------------------------------------------
# Building the model

criterion = nn.CrossEntropyLoss()

word_num = len(allData.dictionary)
my_model = model.RNNModel(env.model, word_num, env.input_size, env.hidden_layers_num,
                       env.layers_num, env.dropouth, env.dropouti,0.1)
###
if env.resume:
    print('Resuming model ...')
    model_load(env.resume)
    optimizer.param_groups[0]['lr'] = env.lr
    my_model.dropouti, my_model.dropouth, my_model.dropout, env.dropoute = env.dropouti, env.dropouth, env.dropout, env.dropoute

###
if env.cuda:
    my_model = my_model.cuda()
    criterion = criterion.cuda()
###
params = list(my_model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################
Statistics = {
    "net_type": env.model,
    "Dropout": [env.dropouti, env.dropouth],
    "epoch": [],
    "train_ppl": [],
    "val_ppl": [],
    "test_ppl": []
}

def evaluate(data_source, batch_size=10):
    # set the mode to eval mode to disable dropout
    my_model.eval()
    total_loss = 0
    hidden = my_model.get_first_hidden(batch_size,env)
    for i in range(0, data_source.size(0) - 1, env.seq_len):
        data, targets = get_batch(data_source, i, env)
        output, hidden = my_model(data, hidden)
        total_loss += len(data) * criterion(output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

lr_start = env.lr
def train(cur_epoch):
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    final_hidden_states = my_model.get_first_hidden(env.batch_size,env)
    batch, i = 0, 0
    seq_len = env.seq_len
    batches_in_epoch = len(train_data) // env.seq_len
    total_batches = batches_in_epoch*env.epochs
    while i < train_data.size(0) - 1 - 1:
        cur_total_batch = (cur_epoch-1)*batches_in_epoch+batch
        optimizer.param_groups[0]['lr'] = lr_start*(math.exp(-cur_total_batch/total_batches))
        my_model.train()
        data, targets = get_batch(train_data, i, env, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        initial_hidden_states = repackage_hidden(final_hidden_states)
        optimizer.zero_grad()

        output, final_hidden_states = my_model(data, initial_hidden_states)

        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if env.clip: torch.nn.utils.clip_grad_norm_(params, env.clip)
        optimizer.step()

        total_loss += loss.data
        #optimizer.param_groups[0]['lr'] = lr2
        if batch % env.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / env.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // env.seq_len, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / env.log_interval, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = env.lr
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
yellow_ticket=0
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if env.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=env.lr, weight_decay=env.wdecay)
    if env.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=env.lr, weight_decay=env.wdecay)

    for epoch in range(1, env.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        print('Done! Calculating losses..')
        train_loss = evaluate(train_data, env.batch_size)
        val_loss = evaluate(val_data, eval_batch_size)
        test_loss = evaluate(test_data, test_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train ppl {:8.2f} | valid ppl {:8.2f} | test ppl {:8.2f}'.format(
                epoch, (time.time() - epoch_start_time), math.exp(train_loss),math.exp(val_loss),math.exp(test_loss)))
        print('-' * 89)

        Statistics["epoch"].append(epoch)
        Statistics["train_ppl"].append(train_loss)
        Statistics["val_ppl"].append(val_loss)
        Statistics["test_ppl"].append(test_loss)

        if (val_loss < stored_loss*0.99) or (yellow_ticket==0):
            if (val_loss >= stored_loss*0.99):
                yellow_ticket = 1
                print('Didn\'t save the model because validation loss was barely decreasing or not at all.',
                      ' Trying for one more epoch..')
            else:
                yellow_ticket = 0
                model_save('model_{}_{:4.2f}.mdl'.format(env.model, env.dropout))
                print('Saved model with the new best validation loss (:')
                stored_loss = val_loss
        else:
            print('Stopped to prevent over-fitting (validation loss is rising). ',
                  'The saved model is for the previous epoch.')
            break


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load('model_{}_{:4.2f}.mdl'.format(env.model, env.dropout))

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | final test ppl = {:8.2f}'.format(math.exp(test_loss)))
print('=' * 89)

f_name = 'stats_{}_{:4.2f}.csv'.format(env.model, env.dropout)
with open(f_name, 'w') as f:
    f.write('Data for Type={} and Dropout={:4.2f},,,\n'.format(env.model, env.dropout))
    f.write('epoch,trail_ppl,val_ppl,test_ppl\n'.format(env.model, env.dropout))
    for i in range(len(Statistics['epoch'])):
        f.write('{:d},{:9.2f},{:9.2f},{:9.2f}\n'.format(Statistics['epoch'][i],Statistics['train_ppl'][i],
                                                        Statistics['val_ppl'][i],Statistics['test_ppl'][i]))
    print("Saved data to %s" % f_name)
