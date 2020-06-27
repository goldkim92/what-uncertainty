import os
import sys
import argparse
from os.path import join

import torch
import torch.nn.functional as F

import data
import models

# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_number', type=str, default='0')
parser.add_argument('--arch', type=str, default='resnet_aleo', help='resnet or resnet_aleo')
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=100)
parser.add_argument('--mc', type=int, default=20, help='number of samples for MC integration')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs',     type=int, default=90)
parser.add_argument('--runs_dir', type=str, default='runs', help='save directory')
# parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

args.runs_dir = join(args.runs_dir, args.arch)
if not os.path.exists(args.runs_dir):
    os.makedirs(args.runs_dir)

def write_log(string):
    with open(join(args.runs_dir,'log.log'), 'a') as lf:
        sys.stdout = lf
        print(string)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_acc = 0.


# ===========================================================
# main
# ===========================================================
if __name__ == "__main__":
    
    ''' Data loader '''
    train_loader, test_loader = data.get_cifar100_loader(args.bs)


    ''' Build model '''
    if args.arch == 'resnet_aleo':
        model = models.ResNet50_modified(args.n_classes, args.mc)
        criterion = F.nll_loss
    elif args.arch == 'resnet':
        model = models.ResNet50()
        criterion = F.cross_entropy
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.1)


    ''' Train '''
    def train(epoch):
        model.train()
        train_loss = 0.
        correct = 0
        total = 0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logprob,_,_ = model(inputs)
            loss = criterion(logprob, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().detach().item()
            total += inputs.size(0)
            correct += logprob.max(1)[1].eq(targets).sum().item()

        train_loss /= total
        correct /= total
        write_log('Epoch {:03d} | Loss: {:.04f} | Acc: {:.03f}'.format(epoch+1, train_loss, correct))


    ''' Test '''
    def test(epoch):
        global best_acc
        model.eval()
        test_loss = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                logprob,_,_ = model(inputs)
                loss = criterion(logprob, targets)

                test_loss += loss.cpu().detach().item()
                total += inputs.size(0)
                correct += logprob.max(1)[1].eq(targets).sum().item()

        test_loss /= total
        correct /= total
        write_log('Epoch {:03d} | Loss: {:.04f} | Acc: {:.03f}'.format(epoch+1, test_loss, correct))

        # Save checkpoint
        acc = 100. * correct
        if acc > best_acc:
            write_log('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, join(args.runs_dir, 'model_ckpt.pth'))
            best_acc = acc


    ''' Run '''
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
        write_log('\n')

