# MNIST train script for HORN
# Felix Effenberger, July 21, 2023

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision

from model import HORN

# command line arguments
parser = argparse.ArgumentParser(description='HORN training script')
parser.add_argument('--num-hidden', type=int, default=32, help='number of units')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--shuffle', action = 'store_true', help='whether to shuffle stimulus time steps')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h (default: 1)')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--omega', type=float, default=0.224, help='natural frequency omega') # 2 * pi / 28 for sMNIST
parser.add_argument('--gamma', type=float, default=0.01, help='damping coefficient gamma')

args = parser.parse_args()
print(args)

# fix seed
torch.manual_seed(args.seed)

# sMNIST as 1-dim time series
dim_input = 1

# 10 MNIST classes
dim_output = 10

# batch size of the test set
batch_size_train = args.batch_size
batch_size_test = 1000

# to shuffle mnist digits
if args.shuffle:
    perm = torch.randperm(784)

# load dataset
size_validation = 1000 # size of validation dataset
train_set = torchvision.datasets.MNIST(root='data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.MNIST(root='data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - size_validation, size_validation])

# data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size_test, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

# instantiate homogeneous HORN model
# see dynamics.py for an example of a heterogeneous HORN
model = HORN(dim_input, args.num_hidden, dim_output, args.h, args.alpha, args.omega, args.gamma)

# can set input and recurrent weights
# model.i2h.weight = ... # n vector
# model.i2h.bias = ... # n vector
# model.h2h.weight = ... # n x n matrix
# model.h2h.bias = ... # n vector

# bce loss and optimizer for training
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

# make output directory and open log file
if not os.path.exists('out'):
    os.makedirs('out')

fh_log = open('out/log.txt', 'a')

# run inference on test set
def evaluate_model(data_loader, epoch = None, batch = None):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        # loop over batches in data loader
        for i, (images, labels) in enumerate(data_loader):
            # reshape batch
            images = images.reshape(batch_size_test, 1, 784)
            images = images.permute(2, 0, 1)

            if args.shuffle:
                images = images[perm, :, :]

            # run model inference - record true returns dynamics
            output = model(images, record = True)
            prediction = output['output']

            # compute loss + number of correct predictions
            test_loss += loss(prediction, labels).item()
            pred_label = prediction.data.max(1, keepdim=True)[1]
            correct += pred_label.eq(labels.data.view_as(pred_label)).sum()

            if i == 0 and not epoch is None and batch is None:
                # plot unit dynamics (amplitudes) for one sample
                plt.figure()

                # loop over all units
                for i in range(args.n_hidden):
                    plt.plot(output['rec_x_t'][0, :, i])
                plt.title(f'epoch {epoch}, batch {batch}')
                plt.xlabel('time')
                plt.ylabel('amplitude')
                plt.savefig(f'out/dynamics_epoch{epoch:02d}_batch{batch:03d}.png')

                plt.close()

    # compute loss and accuracy
    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

# training loop
best_eval = 0.
for epoch in tqdm(range(args.epochs), total = args.epochs):
    tqdm.write(f'epoch {epoch}')

    # loop over batches for one epoch
    for batch, (images, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
        # set model into train mode
        model.train()

        # reshape samples
        images = images.reshape(-1, 1, 784)

        # dimensions: time x batch x 1
        images = images.permute(2, 0, 1)

        if args.shuffle:
            # shuffle if requested
            images = images[perm, :, :]

        # zero gradients
        optimizer.zero_grad()

        # predict
        output = model(images)
        prediction = output['output']

        # compute loss
        train_loss = loss(prediction, labels)

        # compute gradients
        train_loss.backward()

        # update parameters
        optimizer.step()

        if batch % 100 == 0:
            test_acc = evaluate_model(test_loader, epoch, batch)
            tqdm.write(f'epoch {epoch} batch {batch}: test acc {test_acc:.2f}')

    # compute validation and test accuracy
    valid_acc = evaluate_model(valid_loader)
    test_acc = evaluate_model(test_loader, epoch, batch)
    if valid_acc > best_eval:
        best_eval = valid_acc
        final_test_acc = test_acc

    # log accuracy
    msg = f'val: {valid_acc:.4f}, test: {test_acc:.4f}'
    fh_log.write(msg + '\n')
    tqdm.write(msg)

    # save checkpoint
    fn_checkpoint = f'out/epoch{epoch:02d}.pt'
    torch.save(model.state_dict(), fn_checkpoint)
    tqdm.write(f'wrote checkpoint {fn_checkpoint}')

# all done
msg = f'best test: {final_test_acc:.2f}'
fh_log.write(msg + '\n')
fh_log.close()
print(msg)
