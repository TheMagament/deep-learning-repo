# ----------------------------------------------------------------------------------
# Imports
import time
import math
import torch
import torch.nn as nn
import os
import DataHandler
import model
from collections import namedtuple
from utils import batchify, get_batch, repackage_hidden

# ----------------------------------------------------------------------------------
# Global paramerets
env = namedtuple('env',[])
env.data = 'data'
env.model = 'LSTM'
env.input_size = 200
env.hidden_layers_num = 200
env.layers_num = 2
env.lr = 30
env.clip = 0.5
env.epochs = 10
env.batch_size = 20
env.seq_len = 10
env.dropouth = 0.2
env.dropouti = 0.2
env.wdrop = 0
env.seed = 141
env.nonmono = 5
env.log_interval = 200
env.save = 'PTB.pt'
env.alpha = 2
env.beta = 1
env.wdecay = 1.2e-6
env.resume = ''
env.optimizer = 'sgd'
env.when = [-1]

env.tied = True
env.cuda = torch.cuda.is_available()

# ----------------------------------------------------------------------------------
# Data loading

# Functions for saving and loading the model
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

# Convert the data text files into something that is easy to work with
f_name = 'encoded_data'
if os.path.exists(f_name):
    print('Loading cached dataset...')
    allData = torch.load(f_name)
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
model = model.RNNModel(env.model, word_num, env.input_size, env.hidden_layers_num, env.layers_num, env.dropouth, env.dropouti)
###
if env.resume:
    print('Resuming model ...')
    model_load(env.resume)
    optimizer.param_groups[0]['lr'] = env.lr
    model.dropouti, model.dropouth, model.dropout, env.dropoute = env.dropouti, env.dropouth, env.dropout, env.dropoute

###
if env.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # set the mode to eval mode to disable dropout
    model.eval()
    total_loss = 0
    hidden = model.get_first_hidden(batch_size,env)
    for i in range(0, data_source.size(0) - 1, env.seq_len):
        data, targets = get_batch(data_source, i, env)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

lr_start = env.lr
def train(cur_epoch):
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    final_hidden_states = model.get_first_hidden(env.batch_size,env)
    batch, i = 0, 0
    seq_len = env.seq_len
    batches_in_epoch = len(train_data) // env.seq_len
    total_batches = batches_in_epoch*env.epochs
    while i < train_data.size(0) - 1 - 1:
        cur_total_batch = cur_epoch*batches_in_epoch+batch
        optimizer.param_groups[0]['lr'] = lr_start*(math.exp(-3*cur_total_batch/total_batches))
        model.train()
        data, targets = get_batch(train_data, i, env, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        initial_hidden_states = repackage_hidden(final_hidden_states)
        optimizer.zero_grad()

        output, final_hidden_states = model(data, initial_hidden_states)

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
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
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
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ppl {:8.2f}'.format(
                    epoch, (time.time() - epoch_start_time), math.exp(val_loss2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(env.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ppl {:8.2f}'.format(
              epoch, (time.time() - epoch_start_time), math.exp(val_loss)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(env.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if env.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>env.nonmono and val_loss > min(best_val_loss[:-env.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=env.lr, t0=0, lambd=0., weight_decay=env.wdecay)

            if epoch in env.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(env.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(env.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test ppl {:8.2f}'.format(math.exp(test_loss)))
print('=' * 89)
