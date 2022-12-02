import argparse
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter # TensorBoard support
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from IPython import display
# import modules to build RunBuilder and RunManager helper classes
from collections  import OrderedDict
import os
import errno
import sys
sys.path.insert(1, '/lcrc/project/FastBayes/sanket_bnn/SS_IG/')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from Gauss_linear_layers import Gauss_layer

parser = argparse.ArgumentParser(description='MLP Experiments')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--results_path', default='./results_mlp/', type = str, help = 'base path for saving result')

# Data setting
parser.add_argument('--data', type=str, default='mnist')

parser.add_argument('--hidden_dim', default = 400, type = int, help = 'Number of hidden laye nodes')

# Training Setting
parser.add_argument('--nepoch', default = 1200, type = int, help = 'total number of training epochs')
parser.add_argument('--init_lr', default = 0.001, type = float, help = 'initial learning rate')
parser.add_argument('--batch_train', default = 1024, type = int, help = 'batch size for training')
parser.add_argument('--num_MC_test', default = 10, type = int, help = 'Number of MC samples for testing')

# Optimizer Setting
# parser.add_argument('--clip', default = 1, type = float, help = 'Gradient clipping value')

# Prior Setting
parser.add_argument('--sigma_0', default = 1, type = float, help = 'sigma_0 in prior')

args = parser.parse_args()

writer = SummaryWriter('runs/Gaussian_MLP'+str(args.hidden_dim)+'_'+str(args.data)+
                                '_relu_sig0_'+str(args.sigma_0)+'_lr_'+str(args.init_lr)+
                                '_batch_'+ str(args.batch_train)+'_total_epochs_'+str(args.nepoch))
                                        
#### sparsefunc file content
class SFunc(nn.Module):
    """
        Our BNN
    """
    def __init__(self, hidden_dim1, hidden_dim2, target_dim, sigma_0):

        # initialize the network using the MLP layer
        super().__init__()

        # Spike-and-slab Gaussian node selection
        self.l1 = Gauss_layer(28*28, hidden_dim1, sigma_0=sigma_0)
        self.l2 = Gauss_layer(hidden_dim1, hidden_dim2, sigma_0=sigma_0)
        self.l3 = Gauss_layer(hidden_dim2, target_dim, sigma_0=sigma_0)

    def forward(self, X):
        """
            output of the BNN for one Monte Carlo sample
            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """        
        output = F.relu(self.l1(X.reshape(-1,28*28)))
        output = F.relu(self.l2(output))
        output = self.l3(output)        
        return output


class RunManager():
    def __init__(self):
        # tracking every epoch count, loss, accuracy, time
        self.epoch_start_time = None
        self.run_data = []

    def begin_run(self):
        self.run_start_time = time.time()

    def begin_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self,epoch,train_loss,train_accuracy,test_loss,
                    test_accuracy,learning_rate,batch_size):
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results["epoch"] = epoch
        results["Train loss"] = train_loss    #loss
        results["Test loss"] = test_loss
        results["Train Accuracy"] = train_accuracy   #accuracy
        results["Test Accuracy"] = test_accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        results["lr"] = learning_rate
        results["batch_size"] = batch_size

        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient = 'columns')
        # display epoch information and show progress
        display.clear_output(wait=True)
        display.display(df)
  
    def save(self, Path, fileName):
        pd.DataFrame.from_dict(
            self.run_data, 
            orient = 'columns',
        ).to_csv(f'{Path+fileName}.csv')

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(args)

    if args.data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x * 255. / 126.)])

        train_set = datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transform)
        target_dim = 10

    elif args.data == 'fashion_mnist':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        train_set = datasets.FashionMNIST(root = './data/FashionMNIST', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root = './data/FashionMNIST', train=False, download=True, transform=transform)                              
        target_dim = 10

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,num_workers=8,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=8,pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m = RunManager()
    m.begin_run()

    hidden_dim1 = args.hidden_dim
    hidden_dim2 = args.hidden_dim

    sigma_0 = torch.as_tensor(args.sigma_0).to(device)
    num_MC_test = args.num_MC_test

    loss_func = nn.CrossEntropyLoss().to(device)

    net = SFunc(hidden_dim1, hidden_dim2, target_dim, sigma_0).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    learning_rate = args.init_lr

    PATH = args.results_path + args.data + '/Gaussian/MLP' +str(args.hidden_dim) + '/'
    print(PATH)
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = args.nepoch
    train_Loss = []
    train_Accuracy = []
    test_Loss = []
    test_Accuracy = []

    NTrain = len(train_loader.dataset)

    for epoch in range(num_epochs):
        print('----------Epoch {}----------------'.format(epoch))
        m.begin_epoch()
        net.train()
        train_loss = 0.
        correct_train = 0

        for batch in train_loader:
            images, labels = batch[0].to(device), batch[1].to(device)

            output = net(images)
            nll_train = loss_func(output, labels)         
            kl_train = net.l1.kl+net.l2.kl+net.l3.kl
            loss = nll_train + kl_train.div(NTrain)          

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            train_loss += nll_train.mul(images.shape[0]).item()
            correct_train += output.data.argmax(1).eq(labels.data).sum().item()

        with torch.no_grad():
            net.eval()
            train_accuracy = correct_train / len(train_set)
            train_loss = train_loss / len(train_set)

            train_Loss.append(train_loss)
            train_Accuracy.append(train_accuracy)

            test_loss = 0
            prev = 0
            param_no_list = []
            flop_list = []
            outputs = torch.zeros(num_MC_test, len(test_set), target_dim).to(device)
            final_labels = torch.empty(len(test_set)).to(device)   
            for _ , (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device) 
                final_labels[prev:prev+labels.shape[0]] = labels    
                nll_test_comp = 0        
                for it in range(num_MC_test):       
                    outputs[it,prev:prev+labels.shape[0],:] = net(images)                    
                    nll_test_comp += loss_func(outputs[it,prev:prev+labels.shape[0],:], labels)      
                    param_overall = 0.
                    flops_overall = 0.
                    for _ , module in net.named_modules():            
                        if isinstance(module, (Gauss_layer)):
                            if module.v is not None:
                                param_overall += (module.input_dim+1)*module.output_dim
                                flops_overall += (module.input_dim+1)*module.output_dim
                            else:
                                param_overall += module.input_dim*module.output_dim
                                flops_overall += module.input_dim*module.output_dim
                    param_no_list.append(param_overall)
                    flop_list.append(flops_overall)
                    print(f"Total num para= {param_overall}, and flops is {flops_overall}")
                test_loss += nll_test_comp.div(num_MC_test).mul(images.shape[0]).item()
                prev += labels.shape[0] 
            test_loss = test_loss / len(test_set)
            output_mean = outputs.mean(dim=0)
            test_accuracy = output_mean.data.argmax(1).eq(final_labels.data).sum().div(len(test_set)).item() 

            flops_pruned_val = np.median(flop_list)
            param_pruned_val = np.median(param_no_list)
            print(f"Total num para= {param_pruned_val}, and flops is {flops_pruned_val}")

            test_Loss.append(test_loss)
            test_Accuracy.append(test_accuracy)

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {}'.format(
                    epoch, train_loss, train_accuracy, test_loss, test_accuracy))

        m.end_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                    learning_rate, args.batch_train)

    print('Finished Training')
    writer.close()
    
    m.save(PATH,'results_Gaussian_MLP'+str(args.hidden_dim)+'_'+str(args.data)+
                                '_relu_sig0_'+str(args.sigma_0)+'_lr_'+str(args.init_lr)+
                                '_batch_'+ str(args.batch_train)+'_total_epochs_'+str(args.nepoch))
    torch.save(net.state_dict(), PATH + 'Gaussian_MLP'+str(args.hidden_dim)+'_'+str(args.data)+
                                '_relu_sig0_'+str(args.sigma_0)+'_lr_'+str(args.init_lr)+
                                '_batch_'+ str(args.batch_train)+'_total_epochs_'+str(args.nepoch)+'.pt')
        
    # plt.plot(range(num_epochs), train_Loss, 'b', label='Train')                
    # plt.plot(range(num_epochs), test_Loss, 'orange', label='Test')
    # plt.title('Gaussian: Train-Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # # plt.ylim([0.55, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"Gaussian_MLP400_MNIST_loss_relu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), train_Accuracy, 'b', label='Train')
    # plt.plot(range(num_epochs), test_Accuracy, 'orange', label='Test')
    # plt.title('Gaussian: Train-Test Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # # plt.ylim([0.55, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"Gaussian_MLP400_MNIST_accuracy_relu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), sparsity_mlp1, 'r', label='MLP Layer 1')
    # plt.plot(range(num_epochs), sparsity_mlp2, 'b', label='MLP Layer 2')
    # plt.title('Gaussian: Node Sparsity')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sparsity')
    # # plt.ylim([0.65, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"Gaussian_MLP400_MNIST_sparsity_node_relu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), Edge_sparsity, 'g')
    # plt.title('Gaussian: Edge Sparsity')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sparsity')
    # # plt.ylim([0.65, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"Gaussian_MLP400_MNIST_sparsity_edge_relu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

if __name__ == '__main__':
    main()