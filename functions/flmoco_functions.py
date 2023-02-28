'''
SFL-mocoV1 basic functionality, wrap in a class. 

Can be extended to real-applcations if communication protocol is considered.

To understand how the training works in this implementation of SFL. We provide a tutorial in __main__ function:

Refer to He et al. Momentum Contrast for Unsupervised Visual Representation Learning for technical details.

'''

import torch
import copy
import math
import torch.nn as nn
from functions.base_funtions import base_simulator, create_base_instance
import torchvision.transforms as transforms
import torch.nn.functional as F
from models.resnet import init_weights
from utils import AverageMeter, accuracy


class flmoco_simulator(base_simulator):
    def __init__(self, model, criterion, train_loader, test_loader, args) -> None:
        super().__init__(model, criterion, train_loader, test_loader, args)
        if self.model.get_num_of_cloud_layer() != 0:
            self.model.resplit(0)

        # Create instances
        self.c_instance_list = []
        for i in range(args.num_client):
            self.c_instance_list.append(create_flmococlient_instance(self.model.local_list[i], criterion, int(args.batch_size * args.client_sample_ratio * self.num_client), args))
        
        # Set optimizer
        self.c_optimizer_list = [None for i in range(args.num_client)]
        for i in range(args.num_client):
            self.c_optimizer_list[i] = torch.optim.SGD(list(self.c_instance_list[i].model.parameters()), lr=args.c_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Set optimizer scheduler
        milestones = [60, 120, 160]
        self.c_scheduler_list = [None for i in range(args.num_client)]
        if args.cos:
            for i in range(args.num_client):
                self.c_scheduler_list[i] = torch.optim.lr_scheduler.CosineAnnealingLR(self.c_optimizer_list[i], self.num_epoch)  # learning rate decay
        else:
            for i in range(args.num_client):
                self.c_scheduler_list[i] = torch.optim.lr_scheduler.MultiStepLR(self.c_optimizer_list[i], milestones=milestones, gamma=0.2)  # learning rate decay

        # Set augmentation
        self.K_dim = args.K_dim
    
    def linear_eval(self, memloader, num_epochs = 100, lr = 3.0): # Use linear evaluation
        """
        Run Linear evaluation
        """
        self.cuda()
        self.eval()  #set to eval mode
        criterion = nn.CrossEntropyLoss()

        if self.model.get_num_of_cloud_layer() == 0:
            self.model.resplit(5)
        self.model.unmerge_classifier_cloud()

        classifier_list = [nn.Linear(512 * self.model.expansion, self.num_class)]
        self.model.classifier = nn.Sequential(*classifier_list)

        self.model.classifier.apply(init_weights)

        linear_optimizer = torch.optim.SGD(list(self.model.classifier.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
        linear_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(linear_optimizer, num_epochs)  # learning rate decay 

        self.model.classifier.cuda()
        self.model.classifier.train()
        
        best_avg_accu = 0.0
        # Train the linear layer
        for epoch in range(num_epochs):
            for input, label in memloader[0]:
                linear_optimizer.zero_grad()
                input = input.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model.local_list[0](input)
                    output = self.model.cloud(output)
                    output = F.avg_pool2d(output, 4)
                    output = output.view(output.size(0), -1)
                output = self.model.classifier(output.detach())
                loss = criterion(output, label)
                # loss = loss_xent(output, label)
                loss.backward()
                linear_optimizer.step()
                linear_scheduler.step()
            
            """
            Run validation
            """
            top1 = AverageMeter()
            
            self.model.classifier.eval()

            for input, target in self.validate_loader:
                input = input.cuda()
                target = target.cuda()
                with torch.no_grad():
                    output = self.model(input)
                prec1 = accuracy(output.data, target)[0]
                top1.update(prec1.item(), input.size(0))
            self.model.classifier.train()
            avg_accu = top1.avg
            if avg_accu > best_avg_accu:
                best_avg_accu = avg_accu
            print(f"Epoch: {epoch}, linear eval accuracy - current: {avg_accu:.2f}, best: {best_avg_accu:.2f}")
        self.train()  #set back to train mode
        return best_avg_accu


    def semisupervise_eval(self, memloader, num_epochs = 100, lr = 3.0): # Use semi-supervised learning as evaluation
        """
        Run Linear evaluation
        """
        self.cuda()
        self.eval()  #set to eval mode
        criterion = nn.CrossEntropyLoss()

        if self.model.get_num_of_cloud_layer() == 0:
            self.model.resplit(5)
        self.model.unmerge_classifier_cloud()

        classifier_list = [nn.Linear(512 * self.model.expansion, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(True),
                            nn.Linear(512, self.num_class)]
        self.model.classifier = nn.Sequential(*classifier_list)

        self.model.classifier.apply(init_weights)

        # linear_optimizer = torch.optim.SGD(list(self.model.classifier.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)
        linear_optimizer = torch.optim.Adam(list(self.model.classifier.parameters()), lr=1e-3) # as in divergence-aware
        milestones = [int(0.6*num_epochs), int(0.8*num_epochs)]
        linear_scheduler = torch.optim.lr_scheduler.MultiStepLR(linear_optimizer, milestones=milestones, gamma=0.1)  # learning rate decay 

        self.model.classifier.cuda()
        self.model.classifier.train()
        
        best_avg_accu = 0.0
        # Train the linear layer
        for epoch in range(num_epochs):
            for input, label in memloader[0]:
                linear_optimizer.zero_grad()
                input = input.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model.local_list[0](input)
                    output = self.model.cloud(output)
                    output = F.avg_pool2d(output, 4)
                    output = output.view(output.size(0), -1)
                output = self.model.classifier(output.detach())
                loss = criterion(output, label)
                # loss = loss_xent(output, label)
                loss.backward()
                linear_optimizer.step()
                linear_scheduler.step()
            
            """
            Run validation
            """
            top1 = AverageMeter()
            
            self.model.classifier.eval()

            for input, target in self.validate_loader:
                input = input.cuda()
                target = target.cuda()
                with torch.no_grad():
                    output = self.model(input)
                prec1 = accuracy(output.data, target)[0]
                top1.update(prec1.item(), input.size(0))
            self.model.classifier.train()
            avg_accu = top1.avg
            if avg_accu > best_avg_accu:
                best_avg_accu = avg_accu
            print(f"Epoch: {epoch}, linear eval accuracy - current: {avg_accu:.2f}, best: {best_avg_accu:.2f}")
        self.train()  #set back to train mode
        return best_avg_accu

    def knn_eval(self, memloader): # Use linear evaluation
        if self.c_instance_list:
            self.c_instance_list[0].cuda()
        # test using a knn monitor
        def test():
            self.eval()
            classes = self.num_class
            total_top1, total_top5, total_num, feature_bank, feature_labels = 0.0, 0.0, 0, [], []
            with torch.no_grad():
                # generate feature bank
                for data, target in memloader[0]:
                    feature = self.model(data.cuda(non_blocking=True))
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
                    feature_labels.append(target)
                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().cuda()
                # [N]
                feature_labels = torch.cat(feature_labels, dim=0).contiguous().cuda()
                # feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
                # loop test data to predict the label by weighted knn search
                for data, target in self.validate_loader:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    feature = self.model(data)
                    feature = F.normalize(feature, dim=1)
                    
                    pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

                    total_num += data.size(0)
                    total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                    # print('KNN Test: Acc@1:{:.2f}%'.format(total_top1 / total_num * 100))

            return total_top1 / total_num * 100

        # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
        # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
        def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            return pred_labels
        test_acc_1 = test()
        self.train() #set back to train
        return test_acc_1

class create_flmococlient_instance(create_base_instance):
    def __init__(self, model, criterion, batch_size, args) -> None:
        super().__init__(model)
        self.criterion = criterion
        self.batch_size = batch_size
        self.t_model = copy.deepcopy(model)
        self.symmetric = args.symmetric

        for param_t in self.t_model.parameters():
            param_t.requires_grad = False  # not update by gradient

        self.K = args.K
        self.T = args.T
        self.queue = torch.randn(args.K_dim, args.K).cuda()
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.queue_ptr = torch.zeros(1, dtype=torch.long)
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        output = self.model(input)
        return output


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:] = keys.T[:, :self.K - ptr]
            self.queue[:, 0:(batch_size + ptr - self.K)] = keys.T[:, self.K - ptr:]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_moving_average(self):
        tau = 0.99 # default value in moco
        for online, target in zip(self.model.parameters(), self.t_model.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]


    def contrastive_loss(self, query, pkey):
        query_out = self.model(query)

        query_out = nn.functional.normalize(query_out, dim = 1)

        with torch.no_grad():  # no gradient to keys

            pkey_, idx_unshuffle = self._batch_shuffle_single_gpu(pkey)

            pkey_out = self.t_model(pkey_)

            pkey_out = nn.functional.normalize(pkey_out, dim = 1).detach()

            pkey_out = self._batch_unshuffle_single_gpu(pkey_out, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [query_out, pkey_out]).unsqueeze(-1)
        
        l_neg = torch.einsum('nc,ck->nk', [query_out, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = self.criterion(logits, labels)

        accu = accuracy(logits, labels)

        return loss, accu, query_out, pkey_out

    def compute(self, query, pkey):

        self.update_moving_average()

        if self.symmetric:
            loss12, accu, q1, k2 = self.contrastive_loss(query, pkey)
            loss21, accu, q2, k1 = self.contrastive_loss(pkey, query)
            loss = loss12 + loss21
            pkey_out = torch.cat([k1, k2], dim = 0)
        else:
            loss, accu, query_out, pkey_out = self.contrastive_loss(query, pkey)

        self._dequeue_and_enqueue(pkey_out)

        error = loss.detach().cpu().numpy()
        
        # loss.backward(retain_graph = True)
        loss.backward()

        return error, accu[0]
    
    def cuda(self):
        self.model.cuda()
        self.t_model.cuda()
    
    def cpu(self):
        self.model.cpu()
        self.t_model.cpu()
