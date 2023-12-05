import argparse
import datetime
import math
import os
import time
import matplotlib 
 
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import data
from model.rnn import RNN

parser = argparse.ArgumentParser(description='PyTorch Language Model - RNN Basic')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
# you can increase the seqence length to see how well the model works when capturing long-term dependencies
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--clipping_theta', type=float, default=1e-2, help='Clipping Theta')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
parser.add_argument('--embed_input_size', type=int, default=1000, help='embed input size')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden_size')
parser.add_argument('--layer_num', type=int, default=1, help='layer_num')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--model', type=str, default='gru', help='gru|lstm')

# feel free to add some other arguments
args = parser.parse_args()
clipping_theta = args.clipping_theta

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("data", batch_size, args.max_sql)

voc_size = len(data_loader.vocabulary)
embed_input_size = args.embed_input_size
hidden_size = args.hidden_size
layer_num = args.layer_num

# WRITE CODE HERE within two '#' bar                                                           #
# Build model, optimizer and so on                                                             #
################################################################################################
# model
net = RNN(nvoc=voc_size, ninput=embed_input_size, nhid=hidden_size, nlayers=layer_num, device=device, model=args.model)

# Cross Entropy Loss
lr = args.lr
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)


def log(s):
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} --- {s}")


################################################################################################
# WRITE CODE HERE within two '#' bar                                                           #
# Evaluation Function                                                                          #
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       #
################################################################################################
def evaluate(rnn_net, _device):
    data_loader.set_valid()
    rnn_net.eval()
    end_flag = False
    _state = None

    correct, tot_loss, unit_count = 0, 0, 0
    with torch.no_grad():
        while not end_flag:
            x, y, end_flag = data_loader.get_batch()
            x = x.to(device)
            y = y.to(device)
            _state = None
            y_hat, _state = rnn_net(x, _state)

            y_hat = y_hat.view(y_hat.shape[0] * y_hat.shape[1], -1)
            loss = criterion(y_hat, y).mean()

            _, predicted = y_hat.max(1)
            correct += predicted.eq(y).sum().item()

            tot_loss += loss * y.numel()
            unit_count += y.numel()
        return math.exp(tot_loss / unit_count), (100.0 * correct) / unit_count


################################################################################################
# WRITE CODE HERE within two '#' bar                                                           #
# Training Function                                                                            #     
# Calculate the average cross-entropy loss between the prediction and the ground truth word    #
# And then exp(average cross-entropy loss) is perplexity                                       # 
################################################################################################
# 裁剪梯度, from d2l
def grad_clipping(rnn_net, theta):
    if isinstance(rnn_net, nn.Module):
        params = [p for p in rnn_net.parameters() if p.requires_grad]
    else:
        params = rnn_net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


train_acc_curve = []
val_acc_curve = []
train_loss_curve = []
val_loss_curve = []
best_loss = None


def save_plot(name):
    epochs = range(len(train_acc_curve))
    plt.plot(epochs, train_acc_curve, 'b', label="Training Acc")
    plt.plot(epochs, val_acc_curve, 'r', label="Validation Acc")
    plt.title('Training and Validation Acc')
    plt.legend()
   # plt.savefig(os.path.join(args.save_dir, f'acc_{name}.png'))
    plt.savefig('./result3.png')
    plt.close()

    epochs = range(len(train_loss_curve))
    plt.plot(epochs, train_loss_curve, 'b', label="Training PPL")
    plt.plot(epochs, val_loss_curve, 'r', label="Validation PPL")
    plt.title('Training and Validation Perplexity')
    plt.legend()
   # plt.savefig(os.path.join(args.save_dir, f'loss_{name}.png'))
    plt.savefig('./result4.png')
    plt.close()


def train(rnn_net, _device):
    data_loader.set_train()
    rnn_net.train()
    end_flag = False

    correct = 0
    tot_loss, unit_count = 0, 0
    while not end_flag:
        x, y, end_flag = data_loader.get_batch()
        x = x.to(device)
        y = y.to(device)
        _state = None

        y_hat, _state = rnn_net(x, _state)

        y_hat = y_hat.view(y_hat.shape[0] * y_hat.shape[1], -1)
        loss = criterion(y_hat, y).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_clipping(rnn_net, clipping_theta)
        optimizer.step()

        _, predicted = y_hat.max(1)
        correct += predicted.eq(y).sum().item()

        tot_loss += loss * y.numel()
        unit_count += y.numel()
    return math.exp(tot_loss / unit_count), (100.0 * correct) / unit_count


################################################################################################

# WRITE CODE HERE within two '#' bar                                                           #
# Showcase your model with multi-step ahead prediction                                         #
################################################################################################
def predict(my_net):
    prompt = ["The", "player", "progress", "through", "a", "serious", "of"]
    for w in prompt:
        print(w, end=' ')
    voc = []
    for word in prompt:
        voc.append(data_loader.word_id[word])
    prompt = torch.LongTensor(voc).unsqueeze(-1).to(device)
    my_net.eval()
    _state = None
    output,hidden = my_net(prompt, _state)
    for i in range(20):
        last = output[-1, -1]
        idx = torch.argmax(last)
        if data_loader.vocabulary[idx]== '<eos>':
            break
        print(data_loader.vocabulary[idx],end=' ')
        voc.append(idx)
        prompt = torch.LongTensor(voc).unsqueeze(-1).to(device)
        output, hidden = my_net(prompt, hidden)
    
    print()
    


################################################################################################

################################################################################################
# WRITE CODE HERE within two '#' bar                                                           #
# Loop over epochs                                                                             #
################################################################################################
for epoch in range(1, args.epochs + 1):
    train_perplexity, train_acc = train(net, device)
    log(f"epoch {epoch}, train perplexity={train_perplexity}, train acc={train_acc}")
    valid_perplexity, valid_acc = evaluate(net, device)
    log(f"epoch {epoch}, valid perplexity={valid_perplexity}, valid acc={valid_acc}")

    if best_loss is None:
        best_loss = valid_perplexity

    train_acc_curve.append(train_acc)
    val_acc_curve.append(valid_acc)

    train_loss_curve.append(train_perplexity)
    val_loss_curve.append(valid_perplexity)

    if valid_perplexity <= best_loss:
        best_loss = valid_perplexity
        # torch.save(net, os.path.join(args.save_dir, "best_model.pt"))

    torch.save(net.state_dict(), 'temp.pt')
    #save_plot(str(epoch))

model_temp = RNN(nvoc=voc_size, ninput=embed_input_size, nhid=hidden_size, nlayers=layer_num, device=device, model=args.model)
model_temp.load_state_dict(torch.load('temp.pt'))
model_temp.eval()
predict(model_temp)
################################################################################################
