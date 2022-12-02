#!/bin/bash
#SBATCH --job-name=SS_Gaussian_Node_MLP  # create a short name for your job
#SBATCH --account=FASTBAYES
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
##SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN        # send email when job begins
#SBATCH --mail-type=END          # send email when job ends
#SBATCH --mail-user=sjantre@anl.gov

# module purge

# module load anaconda3/2020.11

# export PATH="/gpfs/fs1/soft/swing/manual/anaconda3/2020.11/bin:$PATH"
source activate test1

# MLP - MNIST
# python3 SS_Gaussian_Node_MLP.py --data mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 SS_Gaussian_Edge_MLP.py --data mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 Gaussian_MLP.py --data mnist --nepoch 1200 --init_lr 0.0001 --batch_train 128 --num_MC_test 10 --sigma_0 1

# MLP - FashionMNIST
# python3 SS_Gaussian_Node_MLP.py --data fashion_mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 SS_Gaussian_Edge_MLP.py --data fashion_mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 Gaussian_MLP.py --data fashion_mnist --nepoch 1200 --init_lr 0.0001 --batch_train 128 --num_MC_test 10 --sigma_0 1

# Lenet Caffe - MNIST
# python3 SS_Gaussian_Node_Lenet.py --data mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 SS_Gaussian_Edge_Lenet.py --data mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 Gaussian_Lenet.py --data mnist --nepoch 1200 --init_lr 0.0001 --batch_train 128 --num_MC_test 10 --sigma_0 1

# Lenet Caffe - FashionMNIST
python3 SS_Gaussian_Node_Lenet.py --data fashion_mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 SS_Gaussian_Edge_Lenet.py --data fashion_mnist --nepoch 1200 --init_lr 0.001 --batch_train 1024 --num_MC_test 10 --sigma_0 1
# python3 Gaussian_Lenet.py --data fashion_mnist --nepoch 1200 --init_lr 0.0001 --batch_train 128 --num_MC_test 10 --sigma_0 1

conda deactivate