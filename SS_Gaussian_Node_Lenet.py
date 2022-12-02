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

from Gauss_linear_layers import SSGauss_Node_layer, Gauss_layer
from Gauss_Conv_layers import SSGauss_Node_Conv2d_layer

parser = argparse.ArgumentParser(description='Lenet Caffe Experiments')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--results_path', default='./results_lenet/', type = str, help = 'base path for saving result')

# Data setting
parser.add_argument('--data', type=str, default='mnist')

# Training Setting
parser.add_argument('--nepoch', default = 1200, type = int, help = 'total number of training epochs')
parser.add_argument('--init_lr', default = 0.001, type = float, help = 'initial learning rate')
parser.add_argument('--batch_train', default = 1024, type = int, help = 'batch size for training')
parser.add_argument('--num_MC_test', default = 10, type = int, help = 'Number of MC samples for testing')

# Optimizer Setting
# parser.add_argument('--clip', default = 1, type = float, help = 'Gradient clipping value')

# Prior Setting
parser.add_argument('--sigma_0', default = 1, type = float, help = 'sigma_0 in prior')
parser.add_argument('--temp', default = 0.5, type = float, help = 'temperature')
# parser.add_argument('--gamma_prior', default = 0.0001, type = float, help = 'prior inclusion probaility for filters/nodes')

args = parser.parse_args()

writer = SummaryWriter('runs/SS_Gaussian_Node_FLOPs_Modified_Lenet_Caffe_'+str(args.data)+
                            '_silu_sig0_'+str(args.sigma_0)+'_lr_'+str(args.init_lr)+
                            '_batch_'+ str(args.batch_train)+'_total_epochs_'+str(args.nepoch))

#### sparsefunc file content
class SFunc(nn.Module):
    """
        Our BNN
    """
    def __init__(self, hidden_dim1, hidden_dim2, target_dim, temp, gamma_prior, 
                    gamma_prior_star1, gamma_prior_star2, sigma_0):

        # initialize the network using the MLP layer
        super().__init__()

        # Spike-and-slab Gaussian node selection
        self.conv1 = SSGauss_Node_Conv2d_layer(1, 20, kernel_size=5, stride=1, temp=temp, 
                                        gamma_prior=gamma_prior_star1, sigma_0=sigma_0)
        self.conv2 = SSGauss_Node_Conv2d_layer(20, 50, kernel_size=5, stride=1, temp=temp, 
                                        gamma_prior=gamma_prior_star2, sigma_0=sigma_0)
        self.l1 = SSGauss_Node_layer(4*4*50, hidden_dim1, temp=temp, gamma_prior=gamma_prior_star1, 
                                        sigma_0=sigma_0)
        self.l2 = SSGauss_Node_layer(hidden_dim1, hidden_dim2, temp=temp, gamma_prior=gamma_prior_star2, 
                                        sigma_0=sigma_0)
        self.l3 = Gauss_layer(hidden_dim2, target_dim, sigma_0=sigma_0)

    def forward(self, X):
        """
            output of the BNN for one Monte Carlo sample
            :param X: [batch_size, data_dim]
            :return: [batch_size, target_dim]
        """        
        output = F.silu(F.max_pool2d(self.conv1(X), 2))
        output = F.silu(F.max_pool2d(self.conv2(output), 2))
        output = F.silu(self.l1(output.reshape(-1,4*4*50)))
        output = F.silu(self.l2(output))
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
                    test_accuracy,sparsity_conv1,sparsity_conv2,
                    sparsity_mlp1,sparsity_mlp2,edge_sparsity,
                    param_pruned,flops_ratio,flops_pruned,
                    learning_rate,batch_size):
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
        results["sparsity conv1"] = sparsity_conv1
        results["sparsity conv2"] = sparsity_conv2
        results["sparsity mlp1"] = sparsity_mlp1
        results["sparsity mlp2"] = sparsity_mlp2
        results["edge sparsity"] = edge_sparsity
        results["param pruned"] = param_pruned
        results["flops ratio"] = flops_ratio
        results["flops pruned"] = flops_pruned
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

    data_size = len(train_set)
    data_dim = 4*4*50
    hidden_dim1 = 800
    hidden_dim2 = 500

    L=2
    u_0= ((L+1)**2)*(np.log(data_size) + np.log(L+1) + np.log(data_dim+1) + np.log(hidden_dim1))
    u_1= ((L+1)**2)*(np.log(data_size) + np.log(L+1) + np.log(hidden_dim1+1) + np.log(hidden_dim2))
    u_2= ((L+1)**2)*(np.log(data_size) + np.log(L+1) + np.log(hidden_dim2+1) + np.log(target_dim))
    v_0=(data_dim+1)**2 + np.log(hidden_dim1+1) + np.log(hidden_dim2+1) + L + np.log(hidden_dim1) + \
            np.log(data_dim+1) + np.log(data_size) + np.log(u_0+u_1+u_2)
    v_1=(hidden_dim1+1)**2 + np.log(data_dim+1) + np.log(hidden_dim2+1) + L + np.log(hidden_dim2) + \
            np.log(hidden_dim1+1) + np.log(data_size) + np.log(u_0+u_1+u_2)
    a_hlayer1 = np.log(hidden_dim1) + 0.000000001*(data_dim+1)*v_0
    a_hlayer2 = np.log(hidden_dim2) + 0.000000001*(hidden_dim1+1)*v_1
    gamma_prior_star1 = torch.as_tensor(1/np.exp(a_hlayer1))
    gamma_prior_star2 = torch.as_tensor(1/np.exp(a_hlayer2)) 

    L=L+1
    total = (data_dim+1) * hidden_dim1  + (hidden_dim1+1)* hidden_dim2 + (hidden_dim2+1) * 10
    a = np.log(total) + 0.1*((L+1)*np.log(max(hidden_dim1,hidden_dim2)) + np.log(np.sqrt(data_size)*data_dim))
    lm = 1/np.exp(a)
    gamma_prior = torch.tensor(lm)

    print('lambda_Edge:',gamma_prior.item(),'lambda_Node_1:',gamma_prior_star1.item(),
            ', Lambda_Node_2:', gamma_prior_star2.item())

    gamma_prior = gamma_prior.to(device)
    gamma_prior_star1 = gamma_prior_star1.to(device)
    gamma_prior_star2 = gamma_prior_star2.to(device)

    sigma_0 = torch.as_tensor(args.sigma_0).to(device)
    temp = torch.as_tensor(args.temp).to(device)
    num_MC_test = args.num_MC_test

    loss_func = nn.CrossEntropyLoss().to(device)

    net = SFunc(hidden_dim1, hidden_dim2, target_dim, temp, gamma_prior, 
                        gamma_prior_star1, gamma_prior_star2, sigma_0).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.init_lr)
    learning_rate = args.init_lr

    total_num_para = 1071880.
    total_flops = 2949030.

    PATH = args.results_path + args.data + '/SS_Gauss_Node/'
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
    sparsity_conv1 = []
    sparsity_conv2 = []
    sparsity_mlp1 = []
    sparsity_mlp2 = []
    Edge_sparsity = []
    flops_ratio = []
    flops_pruned = []
    param_pruned = []

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
            kl_train = net.conv1.kl+net.conv2.kl+net.l1.kl+net.l2.kl+net.l3.kl
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
            # param_no_list = []
            # flop_list = []
            outputs = torch.zeros(num_MC_test, len(test_set), target_dim).to(device)
            final_labels = torch.empty(len(test_set)).to(device)   
            for _ , (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device) 
                final_labels[prev:prev+labels.shape[0]] = labels    
                nll_test_comp = 0        
                for it in range(num_MC_test):       
                    outputs[it,prev:prev+labels.shape[0],:] = net(images)                    
                    nll_test_comp += loss_func(outputs[it,prev:prev+labels.shape[0],:], labels)      
                    # param_overall = 0.
                    # flops_overall = 0.
                    # for _ , module in net.named_modules():
                    #     if isinstance(module, (SSGauss_Node_Conv2d_layer)): 
                    #         in_prune_channels = module.in_channels                   
                    #         out_prune_channels = (module.z != 0).sum()
                    #         if module.v is not None:
                    #             param_overall += (module.w != 0).sum() + (module.v != 0).sum()
                    #             flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1]+1)* \
                    #                             (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]* \
                    #                                 (module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                    #         else:
                    #             param_overall += (module.w != 0).sum()
                    #             flops_overall += out_prune_channels*(in_prune_channels*module.kernel_size[0]*module.kernel_size[1])* \
                    #                             (np.floor((module.input_size[2]+2*module.padding[0]-module.dilation[0]* \
                    #                                 (module.kernel_size[0]-1)-1)/module.stride[0]+1))**2/module.groups
                    #     elif isinstance(module, (SSGauss_Node_layer)):
                    #         out_prune_nodes = (module.z != 0).sum()
                    #         if module.v is not None:
                    #             param_overall += (module.w != 0).sum() + (module.v != 0).sum()
                    #             flops_overall += (module.input_dim+1)*out_prune_nodes
                    #         else:
                    #             param_overall += (module.w != 0).sum()
                    #             flops_overall += module.input_dim*out_prune_nodes                   
                    #     elif isinstance(module, (Gauss_layer)):
                    #         if module.v is not None:
                    #             param_overall += (module.input_dim+1)*module.output_dim
                    #             flops_overall += (module.input_dim+1)*module.output_dim
                    #         else:
                    #             param_overall += module.input_dim*module.output_dim
                    #             flops_overall += module.input_dim*module.output_dim
                    # param_no_list.append(param_overall.item())
                    # flop_list.append(flops_overall.item())
                test_loss += nll_test_comp.div(num_MC_test).mul(images.shape[0]).item()
                prev += labels.shape[0] 
            test_loss = test_loss / len(test_set)
            output_mean = outputs.mean(dim=0)
            test_accuracy = output_mean.data.argmax(1).eq(final_labels.data).sum().div(len(test_set)).item() 

            if net.conv1.transposed:
                conv1_w = net.conv1.w.permute(1, 0, 2, 3)   
                conv1_w = conv1_w.view(-1, conv1_w.shape[1]*conv1_w.shape[2]*conv1_w.shape[3]).T
            else:
                conv1_w = net.conv1.w 
                conv1_w = conv1_w.view(-1, conv1_w.shape[1]*conv1_w.shape[2]*conv1_w.shape[3]).T

            if net.conv2.transposed:
                conv2_w = net.conv2.w.permute(1, 0, 2, 3)   
                conv2_w = conv2_w.view(-1, conv2_w.shape[1]*conv2_w.shape[2]*conv2_w.shape[3]).T
            else:
                conv2_w = net.conv2.w 
                conv2_w = conv2_w.view(-1, conv2_w.shape[1]*conv2_w.shape[2]*conv2_w.shape[3]).T

            flops_pruned_val = 0.

            arr1_l = torch.norm(torch.cat((conv1_w,net.conv1.v.expand(1, net.conv1.v.size()[0])),0),1,0)
            one1_l = (arr1_l!=0).float()
            pruned_conv1_val = torch.sum(one1_l)
            sparsity_conv1_val = (pruned_conv1_val/(arr1_l.size()[0])).item()

            flops_pruned_val += pruned_conv1_val*(net.conv1.in_channels*net.conv1.kernel_size[0]*net.conv1.kernel_size[1]+1)* \
                                    (np.floor((net.conv1.input_size[2]+2*net.conv1.padding[0]-net.conv1.dilation[0]* \
                                    (net.conv1.kernel_size[0]-1)-1)/net.conv1.stride[0]+1))**2/net.conv1.groups

            arr2_l = torch.norm(torch.cat((conv2_w,net.conv2.v.expand(1, net.conv2.v.size()[0])),0),1,0)
            one2_l = (arr2_l!=0).float()
            pruned_conv2_val = torch.sum(one2_l)
            sparsity_conv2_val = (pruned_conv2_val/(arr2_l.size()[0])).item()

            flops_pruned_val += pruned_conv2_val*(pruned_conv1_val*net.conv2.kernel_size[0]*net.conv2.kernel_size[1]+1)* \
                                    (np.floor((net.conv2.input_size[2]+2*net.conv2.padding[0]-net.conv2.dilation[0]* \
                                    (net.conv2.kernel_size[0]-1)-1)/net.conv2.stride[0]+1))**2/net.conv2.groups

            arr1_l = torch.norm(torch.cat((net.l1.w,net.l1.v.expand(1, net.l1.v.size()[0])),0),1,0)
            one1_l = (arr1_l!=0).float()
            pruned_mlp1_val = torch.sum(one1_l)
            sparsity_mlp1_val = (pruned_mlp1_val/(arr1_l.size()[0])).item()

            flops_pruned_val += (pruned_conv2_val*4*4+1)*pruned_mlp1_val

            arr2_l = torch.norm(torch.cat((net.l2.w,net.l2.v.expand(1, net.l2.v.size()[0])),0),1,0)
            one2_l = (arr2_l!=0).float()
            pruned_mlp2_val = torch.sum(one2_l)
            sparsity_mlp2_val = (pruned_mlp2_val/(arr2_l.size()[0])).item()

            flops_pruned_val += (pruned_mlp1_val+1)*pruned_mlp2_val + (pruned_mlp2_val+1)*target_dim
            param_pruned_val = (net.conv1.w != 0).sum() + (net.conv1.v != 0).sum() + \
                                (net.conv2.w != 0).sum() + (net.conv2.v != 0).sum() + \
                                (net.l1.w != 0).sum() + (net.l1.v != 0).sum() + \
                                (net.l2.w != 0).sum() + (net.l2.v != 0).sum() + \
                                (net.l3.w != 0).sum() + (net.l3.v != 0).sum()

            # flops_pruned_val = np.median(flop_list)
            # param_pruned_val = np.median(param_no_list)
            sparsity_overall = param_pruned_val/total_num_para
            flops_ratio_val = flops_pruned_val/total_flops  

            test_Loss.append(test_loss)
            test_Accuracy.append(test_accuracy)
            sparsity_conv1.append(sparsity_conv1_val)
            sparsity_conv2.append(sparsity_conv2_val)
            sparsity_mlp1.append(sparsity_mlp1_val)
            sparsity_mlp2.append(sparsity_mlp2_val)
            flops_pruned.append(flops_pruned_val)
            param_pruned.append(param_pruned_val)
            flops_ratio.append(flops_ratio_val)
            Edge_sparsity.append(sparsity_overall)

        writer.add_scalar('data/loss_train', train_loss, epoch)
        writer.add_scalar('data/accuracy_train', train_accuracy, epoch)
        writer.add_scalar('data/loss_test', test_loss, epoch)
        writer.add_scalar('data/accuracy_test', test_accuracy, epoch)
        writer.add_scalar('data/sparsity_edge', sparsity_overall, epoch)
        writer.add_scalar('data/param_pruned', param_pruned_val, epoch)
        writer.add_scalar('data/flops_ratio', flops_ratio_val, epoch)
        writer.add_scalar('data/flops_pruned', flops_pruned_val, epoch)
        writer.add_scalar('data/learning_rate', learning_rate, epoch)
        writer.add_scalar('data/sparsity_conv1', sparsity_conv1_val, epoch)
        writer.add_scalar('data/sparsity_conv2', sparsity_conv2_val, epoch)
        writer.add_scalar('data/sparsity_mlp1', sparsity_mlp1_val, epoch)
        writer.add_scalar('data/sparsity_mlp2', sparsity_mlp2_val, epoch)

        print('Epoch {}, Train Loss: {}, Train Accuracy: {}, Test Loss: {}, Test Accuracy: {},, Sparsity Conv layer-1: {}, \
                Sparsity Conv layer-2: {},Sparsity MLP layer-1: {}, Sparsity MLP layer-2: {} Edge sparsity: {}, \
                , FLOPs ratio: {}, Param pruned: {}, FLOPs pruned: {}'.format(
                    epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                    sparsity_conv1_val, sparsity_conv2_val, sparsity_mlp1_val, sparsity_mlp2_val, 
                    sparsity_overall,flops_ratio_val,param_pruned_val,flops_pruned_val))

        m.end_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy, 
                    sparsity_conv1_val, sparsity_conv2_val, sparsity_mlp1_val, sparsity_mlp2_val,
                    sparsity_overall, param_pruned_val, flops_ratio_val, flops_pruned_val, learning_rate, args.batch_train)

    print('Finished Training')
    writer.close()
    
    m.save(PATH,'results_SS_Gaussian_Node_FLOPs_Modified_Lenet_Caffe_'+str(args.data)+
                                '_silu_sig0_'+str(args.sigma_0)+'_lr_'+str(args.init_lr)+
                                '_batch_'+ str(args.batch_train)+'_total_epochs_'+str(args.nepoch))
    torch.save(net.state_dict(), PATH + 'SS_Gaussian_Node_FLOPs_Modified_Lenet_Caffe_'+str(args.data)+
                                '_silu_sig0_'+str(args.sigma_0)+'_lr_'+str(args.init_lr)+
                                '_batch_'+ str(args.batch_train)+'_total_epochs_'+str(args.nepoch)+'.pt')
        
    # plt.plot(range(num_epochs), train_Loss, 'b', label='Train')                
    # plt.plot(range(num_epochs), test_Loss, 'orange', label='Test')
    # plt.title('SS_Gaussian_Node: Train-Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # # plt.ylim([0.55, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"SS_Gaussian_Node_Lenet_Caffe_MNIST_loss_silu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), train_Accuracy, 'b', label='Train')
    # plt.plot(range(num_epochs), test_Accuracy, 'orange', label='Test')
    # plt.title('SS_Gaussian_Node: Train-Test Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # # plt.ylim([0.55, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"SS_Gaussian_Node_Lenet_Caffe_MNIST_accuracy_silu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), sparsity_conv1, 'orange', label='Conv Layer 1')
    # plt.plot(range(num_epochs), sparsity_conv2, 'green', label='Conv Layer 2')
    # plt.plot(range(num_epochs), sparsity_mlp1, 'r', label='MLP Layer 1')
    # plt.plot(range(num_epochs), sparsity_mlp2, 'b', label='MLP Layer 2')
    # plt.title('SS_Gaussian_Node: Node Sparsity')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sparsity')
    # # plt.ylim([0.65, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"SS_Gaussian_Node_Lenet_Caffe_MNIST_sparsity_node_silu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

    # plt.plot(range(num_epochs), Edge_sparsity, 'g')
    # plt.title('SS_Gaussian_Node: Edge Sparsity')
    # plt.xlabel('Epochs')
    # plt.ylabel('Sparsity')
    # # plt.ylim([0.65, 1.0])
    # plt.legend(loc='upper right', frameon=False)
    # plt.grid(ls='dotted')
    # plt.savefig(os.path.join(PATH,"SS_Gaussian_Node_Lenet_Caffe_MNIST_sparsity_edge_silu_0.001_1024_1200.png"),dpi=300)
    # plt.close()

if __name__ == '__main__':
    main()