##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Main function for this project. """
import os
import argparse
import numpy as np
from trainer.trainer import Trainer
from utils.gpu_tools import occupy_memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--gpu', default='0', help='the index of GPU')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet'])
    parser.add_argument('--data_dir', default='./data/seed_1993_subset_100_imagenet/data', type=str)
    parser.add_argument('--baseline', default='lucir', type=str, choices=['lucir', 'icarl' ,'geodesic'], help='baseline method')
    parser.add_argument('--ckpt_label', type=str, default='exp02', help='the label for the checkpoints')
    parser.add_argument('--ckpt_dir_fg', type=str, default='./base_50.pt', help='the checkpoint file for the 0-th phase')
    parser.add_argument('--resume_fg', action='store_true', help='resume 0-th phase model from the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from the checkpoints')
    parser.add_argument('--resume_with_ebm_training', action='store_true', help='resume from the checkpoints')
    parser.add_argument('--ckpt_loc', type=str, default='/home/joseph/workspace/xLantern/model_checkpoints/1012_014006', help='the checkpoint file 1st to last phase.')
    parser.add_argument('--num_workers', default=8, type=int, help='the number of workers for loading data')
    parser.add_argument('--random_seed', default=1993, type=int, help='random seed')
    parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size for train loader')
    parser.add_argument('--test_batch_size', default=128, type=int, help='the batch size for test loader')
    parser.add_argument('--eval_batch_size', default=128, type=int, help='the batch size for validation loader')
    parser.add_argument('--disable_gpu_occupancy', action='store_true', help='disable GPU occupancy')
    parser.add_argument('--dataset_location', type=str, default='./data', help='location to store the dataset files')


    ### Network architecture parameters
    parser.add_argument('--branch_mode', default='dual', type=str, choices=['dual', 'single'], help='the branch mode for AANets')
    parser.add_argument('--branch_1', default='ss', type=str, choices=['ss', 'fixed', 'free'], help='the network type for the first branch')
    parser.add_argument('--branch_2', default='free', type=str, choices=['ss', 'fixed', 'free'], help='the network type for the second branch')
    parser.add_argument('--imgnet_backbone', default='resnet18', type=str, choices=['resnet18', 'resnet34'], help='network backbone for ImageNet')

    ### Incremental learning parameters
    parser.add_argument('--num_classes', default=100, type=int, help='the total number of classes')
    parser.add_argument('--nb_cl_fg', default=50, type=int, help='the number of classes in the 0-th phase')
    parser.add_argument('--nb_cl', default=10, type=int, help='the number of classes for each phase')
    parser.add_argument('--nb_protos', default=20, type=int, help='the number of exemplars for each class')
    parser.add_argument('--epochs', default=160, type=int, help='the number of epochs')
    parser.add_argument('--dynamic_budget', action='store_true', help='using dynamic budget setting')
    parser.add_argument('--fusion_lr', default=1e-8, type=float, help='the learning rate for the aggregation weights')

    ### General learning parameters
    parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--custom_weight_decay', default=5e-4, type=float, help='weight decay parameter for the optimizer')
    parser.add_argument('--custom_momentum', default=0.9, type=float, help='momentum parameter for the optimizer')
    parser.add_argument('--base_lr1', default=0.1, type=float, help='learning rate for the 0-th phase')
    parser.add_argument('--base_lr2', default=0.1, type=float, help='learning rate for the following phases')

    ### LUCIR parameters
    parser.add_argument('--the_lambda', default=5, type=float, help='lamda for LF')
    parser.add_argument('--dist', default=0.5, type=float, help='dist for margin ranking losses')
    parser.add_argument('--K', default=2, type=int, help='K for margin ranking losses')
    parser.add_argument('--lw_mr', default=1, type=float, help='loss weight for margin ranking losses')

    ### iCaRL parameters
    parser.add_argument('--icarl_beta', default=0.25, type=float, help='beta for iCaRL')
    parser.add_argument('--icarl_T', default=2, type=int, help='T for iCaRL')

    ### EBM Aligner parameters
    parser.add_argument('--evaluate_with_ebm', action='store_true', help='Evaluate with EBM based alignment')
    parser.add_argument('--ebm_latent_dim', default=64, type=int, help='Input latent dimension')
    parser.add_argument('--ebm_n_layers', default=2, type=int, help='Number of layers in the EBM')
    parser.add_argument('--ebm_n_hidden_units', default=64, type=int, help='Number of hidden neurons in each layer of the EBM')
    parser.add_argument('--max_iter', default=1500, type=int, help='Number of iterations to learn the EBM')
    parser.add_argument('--ebm_lr', default=0.0001, type=float, help='Learning-rate of the EBM')
    parser.add_argument('--n_langevin_steps', default=30, type=int, help='Number of Langevin iterations')
    parser.add_argument('--langevin_lr', default=0.1, type=float, help='Learning-rate for Langevin sampling')
    parser.add_argument('--ema_decay', default=0.89, type=float, help='Exponential Moving Average decay')
    parser.add_argument('--log_iter', default=50, type=int, help='Log after each log_iter iterations')
    parser.add_argument('--output_dir', default='/home/joseph/workspace/lantern', type=str, help='Location to save any plots (if any)')
    parser.add_argument('--use_mixup', action='store_true', help='Enable Mixup if the flag is set')
    parser.add_argument('--beta_param_1', default=0.1, type=float, help='Parameter that controls mixup')
    parser.add_argument('--beta_param_2', default=0.1, type=float, help='Second parameter that controls mixup')

    ### Configure standard IL setting (which doesnt assume large base class)
    parser.add_argument('--enable_std_IL_setting', action='store_true', help='Evaluate with EBM based alignment')
    parser.add_argument('--incremental_classes', default=10, type=int, help='the number of classes for each phase')

    ### Geodesic parameters
    parser.add_argument('--lamda', default=5, type=float, help='Lamda for Geodesic')

    the_args = parser.parse_args()

    if the_args.enable_std_IL_setting:
        the_args.nb_cl_fg = the_args.incremental_classes
        the_args.nb_cl = the_args.incremental_classes

    # Check the number of classes, ensure they are reasonable
    assert(the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert(the_args.nb_cl_fg >= the_args.nb_cl)

    # Print the parameters
    print(the_args)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    # Occupy GPU memory in advance
    if the_args.disable_gpu_occupancy:
        occupy_memory(the_args.gpu)
        print('Occupy GPU memory in advance.')

    # Set the trainer and start training
    trainer = Trainer(the_args)
    trainer.train()
