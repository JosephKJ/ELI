##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Class-incremental learning trainer. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import time
import os
import os.path as osp
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
import models.modified_resnet_cifar as modified_resnet_cifar
import models.modified_resnetmtl_cifar as modified_resnetmtl_cifar
import models.modified_resnet as modified_resnet
import models.modified_resnetmtl as modified_resnetmtl
import models.modified_linear as modified_linear
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_accuracy import compute_accuracy
from trainer.incremental_lucir import incremental_train_and_eval as incremental_train_and_eval_lucir
from trainer.incremental_icarl import incremental_train_and_eval as incremental_train_and_eval_icarl
from trainer.incremental_geodesic_plus_lucir import incremental_train_and_eval as incremental_train_and_eval_geodesic
from trainer.zeroth_phase import incremental_train_and_eval_zeroth_phase as incremental_train_and_eval_zeroth_phase
from utils.misc import process_mnemonics
from trainer.base_trainer import BaseTrainer
from utils.ebm_aligner import EBMAligner

import warnings
warnings.filterwarnings('ignore')

class Trainer(BaseTrainer):
    def train(self):
        """The class that contains the code for the class-incremental system.
        This trianer is based on the base_trainer.py in the same folder.
        If you hope to find the source code of the functions used in this trainer, you may find them in base_trainer.py.
        """
        
        # Set tensorboard recorder
        self.train_writer = SummaryWriter(comment=self.save_path)

        # Initial the array to store the accuracies for each phase
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
        top1_acc_with_alignment_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))

        # Load the training and test samples from the dataset
        X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()

        # Initialize the aggregation weights
        self.init_fusion_vars()       

        # Initialize the class order
        order, order_list = self.init_class_order()
        np.random.seed(None)

        # Set empty lists for the data    
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []

        # Initialize the prototypes
        alpha_dr_herding, prototypes = self.init_prototypes(self.dictionary_size, order, X_train_total, Y_train_total)

        # Set the starting iteration
        # We start training the class-incremental learning system from e.g., 50 classes to provide a good initial encoder
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1

        # Set the models and some parameter to None
        # These models and parameters will be assigned in the following phases
        b1_model = None
        ref_model = None
        b2_model = None
        ref_b2_model = None
        the_lambda_mult = None
        task_wise_acc_all_task = []
        task_wise_acc_with_aligner_all_task = []

        for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
            # Create an aligner per iteration
            EBMAlignerObject = EBMAligner
            aligner = EBMAlignerObject(ebm_latent_dim=self.args.ebm_latent_dim,
                                       ebm_n_layers=self.args.ebm_n_layers,
                                       ebm_n_hidden_units=self.args.ebm_n_hidden_units,
                                       max_iter=self.args.max_iter,
                                       ebm_lr=self.args.ebm_lr,
                                       n_langevin_steps=self.args.n_langevin_steps,
                                       langevin_lr=self.args.langevin_lr,
                                       ema_decay=self.args.ema_decay,
                                       log_iter=self.args.log_iter,
                                       output_dir=self.args.output_dir)

            # Initialize models for the current phase
            b1_model, b2_model, ref_model, ref_b2_model, lambda_mult, cur_lambda, last_iter = self.init_current_phase_model(iteration, start_iter, b1_model, b2_model)

            # Initialize datasets for the current phase
            if len(X_valid_cumuls) > 0:
                X_valid_cumul_prev = np.concatenate(X_valid_cumuls)
                Y_valid_cumul_prev = np.concatenate(Y_valid_cumuls)
                map_Y_valid_cumul_prev = np.array([order_list.index(i) for i in Y_valid_cumul_prev])
                prev_valid_loader = self.create_valid_loader(X_valid_cumul_prev, map_Y_valid_cumul_prev)

            if iteration == start_iter:
                X_train_only_new, map_Y_train_only_new, indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                    X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                    X_train, map_Y_train, map_Y_valid_cumul, X_valid_ori, Y_valid_ori = \
                    self.init_current_phase_dataset(iteration, \
                    start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)
            else:
                X_train_only_new, map_Y_train_only_new, indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                    X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                    X_train, map_Y_train, map_Y_valid_cumul, X_protoset, Y_protoset = \
                    self.init_current_phase_dataset(iteration, \
                    start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                    X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)                

            is_start_iteration = (iteration == start_iter)

            # Imprint weights
            if iteration > start_iter:
                b1_model = self.imprint_weights(b1_model, b2_model, iteration, is_start_iteration, X_train, map_Y_train, self.dictionary_size)

            # Update training and test dataloader
            trainloader, testloader = self.update_train_and_valid_loader(X_train, map_Y_train, X_valid_cumul, map_Y_valid_cumul, \
                iteration, start_iter)
            new_data_trainloader = self.create_train_loader(X_train_only_new, map_Y_train_only_new, drop_last=True)

            # Set the names for the checkpoints
            ckp_name = os.path.join(self.args.ckpt_loc, 'iter_' + str(iteration) + '.pt')
            ckp_name_b2 = os.path.join(self.args.ckpt_loc, 'iter_' + str(iteration) + '_b2.pt')
            ckp_name_fusion_vars = os.path.join(self.args.ckpt_loc, 'iter_' + str(iteration) + '_fusion.pt')

            if iteration==start_iter and self.args.resume_fg:
                # Resume the 0-th phase model according to the config
                b1_model = torch.load(self.args.ckpt_dir_fg)
            elif self.args.resume and os.path.exists(ckp_name):
                print('Loading model...')
                # Resume other models according to the config
                b1_model = torch.load(ckp_name)
                if os.path.exists(ckp_name_b2):
                    b2_model = torch.load(ckp_name_b2)
            else:
                # Start training (if we don't resume the models from the checkpoints)
    
                # Set the optimizer
                tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler = self.set_optimizer(iteration, \
                    start_iter, b1_model, ref_model, b2_model, ref_b2_model)     

                if iteration > start_iter:
                    # Training the class-incremental learning system from the 1st phase

                    # Set the balanced dataloader
                    balancedloader = self.gen_balanced_loader(X_train_total, Y_train_total, indices_train_10, X_protoset, Y_protoset, order_list)

                    # Create a dataloader for exemplars. Labels are ignored.
                    exemplar_dataloader = self.create_train_loader(X_protoset, Y_protoset, drop_last=True)
                    exemplar_plus_current_dataloader = self.create_train_loader(X_train, map_Y_train, drop_last=True)

                    # Training the model for different baselines
                    if self.args.resume_with_ebm_training:
                        if os.path.exists(ckp_name):
                            print('Loading model from {}'.format(ckp_name))
                            b1_model = torch.load(ckp_name)
                        if os.path.exists(ckp_name_b2):
                            print('Loading model from {}'.format(ckp_name_b2))
                            b2_model = torch.load(ckp_name_b2)
                        if os.path.exists(ckp_name_fusion_vars):
                            print('Loading fusion vars from {}'.format(ckp_name_fusion_vars))
                            self.fusion_vars = torch.load(ckp_name_fusion_vars)
                    else:
                        if self.args.baseline == 'lucir':
                            b1_model, b2_model = incremental_train_and_eval_lucir(self.args, self.args.epochs, self.fusion_vars, \
                                self.ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, \
                                fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                                X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, balancedloader)
                        elif self.args.baseline == 'icarl':
                            b1_model, b2_model = incremental_train_and_eval_icarl(self.args, self.args.epochs, self.fusion_vars, \
                                self.ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, \
                                fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                                X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda, self.args.dist, self.args.K, self.args.lw_mr, balancedloader, \
                                self.args.icarl_T, self.args.icarl_beta)
                        elif self.args.baseline == 'geodesic':
                            out_features1 = b1_model.fc.fc1.out_features
                            out_features2 = b1_model.fc.fc2.out_features
                            lamda_mult = (out_features1 + out_features2) * 1.0 / (self.args.nb_cl)
                            cur_lamda = self.args.lamda * math.sqrt(lamda_mult)
                            # b1_model = incremental_train_and_eval_geodesic(self.args.epochs, b1_model, ref_model, tg_optimizer,
                            #                                      tg_lr_scheduler, \
                            #                                      trainloader, testloader, \
                            #                                      iteration, start_iter, \
                            #                                      cur_lamda)
                            b1_model, b2_model = incremental_train_and_eval_geodesic(self.args, self.args.epochs,
                                                                                  self.fusion_vars,
                                                                                  self.ref_fusion_vars, b1_model, ref_model,
                                                                                  b2_model, ref_b2_model, tg_optimizer,
                                                                                  tg_lr_scheduler,
                                                                                  fusion_optimizer, fusion_lr_scheduler,
                                                                                  trainloader, testloader, iteration,
                                                                                  start_iter,
                                                                                  X_protoset_cumuls, Y_protoset_cumuls,
                                                                                  order_list, cur_lambda, self.args.dist,
                                                                                  self.args.K, self.args.lw_mr,
                                                                                  balancedloader, lamda=cur_lamda)

                        else:
                            raise ValueError('Please set the correct baseline.')

                    if self.args.evaluate_with_ebm:
                        aligner.learn_ebm(ref_model, b1_model, new_data_trainloader, exemplar_dataloader,
                                          exemplar_plus_current_dataloader, prev_valid_loader, testloader,
                                          iteration * self.args.nb_cl, b2_model=b2_model, ref_b2_model=ref_b2_model,
                                          fusion_vars=self.fusion_vars, ref_fusion_vars=self.ref_fusion_vars,
                                          args=self.args, iteration=iteration, start_iter=start_iter,
                                          use_mixup=self.args.use_mixup)

                else:         
                    # Training the class-incremental learning system from the 0th phase           
                    b1_model = incremental_train_and_eval_zeroth_phase(self.args, self.args.epochs, b1_model, \
                        ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                        cur_lambda, self.args.dist, self.args.K, self.args.lw_mr) 

            # Select the exemplars according to the current model
            X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding = self.set_exemplar_set(b1_model, b2_model, \
                is_start_iteration, iteration, last_iter, order, alpha_dr_herding, prototypes)
            
            # Compute the accuracies for current phase
            top1_acc_list_ori, top1_acc_list_cumul, top1_acc_with_alignment_list_cumul, task_wise_acc, task_wise_acc_with_alignment = self.compute_acc(class_means, order, order_list, b1_model, b2_model, X_protoset_cumuls, Y_protoset_cumuls, \
                X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, top1_acc_list_ori, top1_acc_list_cumul, self.args.evaluate_with_ebm, aligner, top1_acc_with_alignment_list_cumul)

            task_wise_acc_all_task.append(task_wise_acc)
            task_wise_acc_with_aligner_all_task.append(task_wise_acc_with_alignment)

            # Compute the average accuracy
            num_of_testing = iteration - start_iter + 1
            avg_cumul_acc_fc = np.sum(top1_acc_list_cumul[start_iter:,0])/num_of_testing
            avg_cumul_acc_icarl = np.sum(top1_acc_list_cumul[start_iter:,1])/num_of_testing
            print('\nComputing average accuracy...')
            print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
            print("  Average forgetting            :\t\t{:.2f} %".format(self.compute_fgt(task_wise_acc_all_task)))
            # Write the results to the tensorboard
            self.train_writer.add_scalar('avg_acc/fc', float(avg_cumul_acc_fc), iteration)
            self.train_writer.add_scalar('avg_acc/proto', float(avg_cumul_acc_icarl), iteration)

            if self.args.evaluate_with_ebm and not is_start_iteration:
                avg_cumul_acc_fc_with_align = np.sum(top1_acc_with_alignment_list_cumul[start_iter:, 0]) / num_of_testing
                avg_cumul_acc_icarl_with_align = np.sum(top1_acc_with_alignment_list_cumul[start_iter:, 1]) / num_of_testing
                print('Computing average accuracy with alignment...')
                print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc_with_align))
                print("  Average forgetting            :\t\t{:.2f} %".format(self.compute_fgt(task_wise_acc_with_aligner_all_task)))
                # print(task_wise_acc_all_task)
                # print(task_wise_acc_with_aligner_all_task)
                # Write the results to the tensorboard
                self.train_writer.add_scalar('avg_acc_with_align/fc', float(avg_cumul_acc_fc_with_align), iteration)
                self.train_writer.add_scalar('avg_acc_with_align/proto', float(avg_cumul_acc_icarl_with_align), iteration)

            # Save the model
            save_path = os.path.join(self.model_dir, 'iter_' + str(iteration) + '.pt')
            save_path_b2 = os.path.join(self.model_dir, 'iter_' + str(iteration) + '_b2.pt')
            save_path_fusion_vars = os.path.join(self.model_dir, 'iter_' + str(iteration) + '_fusion.pt')
            print('Saving to {}'.format(save_path))
            torch.save(b1_model, save_path)
            torch.save(b2_model, save_path_b2)
            torch.save(self.fusion_vars, save_path_fusion_vars)

        # Save the results and close the tensorboard writer
        torch.save(top1_acc_list_ori, osp.join(self.save_path, 'acc_list_ori.pth'))
        torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'acc_list_cumul.pth'))
        torch.save(top1_acc_with_alignment_list_cumul, osp.join(self.save_path, 'acc_with_alignment_list_cumul.pth'))

        self.train_writer.close()
