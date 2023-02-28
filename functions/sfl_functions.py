'''
SFL basic functionality, wrap in a class. 

Can be extended to real-applcations if communication protocol is considered.

To understand how the training works in this implementation of SFL. We provide a tutorial in __main__ function:

Refer to Thapa et al. https://arxiv.org/abs/2004.12088 for technical details.

'''

import torch
import copy
import math
import torch.nn as nn
import numpy as np
from functions.base_funtions import base_simulator, create_base_instance
class sfl_simulator(base_simulator):
    def __init__(self, model, criterion, train_loader, test_loader, args) -> None:
        super().__init__(model, criterion, train_loader, test_loader, args)

        # Create instances
        self.s_instance = create_sflserver_instance(self.model.cloud, criterion)
        self.c_instance_list = []
        for i in range(args.num_client):
            self.c_instance_list.append(create_sflclient_instance(self.model.local_list[i]))

        # Set optimizer
        self.s_optimizer = torch.optim.SGD(list(self.s_instance.model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.c_optimizer_list = [None for i in range(args.num_client)]
        for i in range(args.num_client):
            self.c_optimizer_list[i] = torch.optim.SGD(list(self.c_instance_list[i].model.parameters()), lr=args.c_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Set optimizer scheduler
        milestones = [60, 120, 160]
        self.c_scheduler_list = [None for i in range(args.num_client)]
        for i in range(args.num_client):
            self.c_scheduler_list[i] = torch.optim.lr_scheduler.MultiStepLR(self.c_optimizer_list[i], milestones=milestones, gamma=0.2)  # learning rate decay
        self.s_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.s_optimizer, milestones=milestones, gamma=0.2)  # learning rate decay

class create_sflserver_instance(create_base_instance):
    def __init__(self, model, criterion) -> None:
        super().__init__(model)
        self.criterion = criterion
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        output = self.model(input)
        return output

    def compute(self, input, label, only_forward = False):
        input.requires_grad=True
        input.retain_grad()
        output = self.model(input)
        loss = self.criterion(output, label)
        error = loss.detach().cpu().numpy()

        if input.grad is not None:
            input.grad.zero_()
        if not only_forward:
            # loss.backward(retain_graph = True)
            loss.backward()
            gradient = input.grad.detach().clone() # get gradient, the -1 is important, since updates are added to the weights in cpp.
        else:
            gradient = None
        return error, gradient

class create_sflclient_instance(create_base_instance):
    def __init__(self, model) -> None:
        super().__init__(model)
        self.output = None
    def __call__(self, input):
        return self.forward(input)

    def forward(self, input): 
        self.output = self.model(input)
        return self.output.detach()

    def backward(self, external_grad):
        if self.output is not None:
            self.output.backward(gradient=external_grad)
            self.output = None

# class loss_based_status():
#     def __init__(self, loss_threshold, warmup_epoch = 10) -> None:
#         '''Initial status upload and download are true'''
#         self.loss = np.Inf
#         self.communicate = True
#         self.threshold = loss_threshold
#         self.epoch_recording = []
#         self.warmup_epoch = warmup_epoch
    
#     def record_loss(self, epoch, loss):
#         if epoch <= self.warmup_epoch: # make sure your epoch starts from 1
#             delta_loss = 0
#         else:
#             delta_loss = self.loss - loss # if loss reduces, this would be greater than 0 
        
#         self.loss = loss
#         if delta_loss > self.threshold: # if delta loss is large, then we use replay tensor to update server
#             self.communicate = False
#         else:
#             self.communicate = True
#             self.epoch_recording.append(epoch)

class loss_based_status():
    def __init__(self, loss_threshold, warmup_epoch = 10) -> None:
        '''Initial status upload and download are true'''
        self.loss = np.Inf
        self.status = "A"
        self.threshold = loss_threshold
        self.epoch_recording = {"A":0, "B":0, "C":0}
        self.warmup_epoch = warmup_epoch
    
    def record_loss(self, epoch, loss):
        if epoch <= self.warmup_epoch: # make sure your epoch starts from 1
            delta_loss = self.threshold
        else:
            delta_loss = max(self.loss - loss, 0) # if loss reduces, this would be greater than 0 
        
        self.loss = loss
        if delta_loss < self.threshold: # if delta loss is small, then we use replay tensor to update server
            if self.status == "A":
                self.status = "B"
            elif self.status == "B":
                self.status = "C"
        else:
            self.status = "A"
        self.epoch_recording[self.status] += 1


def client_backward(sfl_simulator, pool, gradient_dict):
    for i, client_id in enumerate(pool):
        sfl_simulator.c_instance_list[client_id].backward(gradient_dict[i])
        sfl_simulator.c_optimizer_list[client_id].step()
        sfl_simulator.c_scheduler_list[client_id].step()

if __name__ == '__main__':
    '''This is a tutorial on how to use them'''
    def test_init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.constant_(m.weight, 1.0)
            if m.bias is not None: 
                m.bias.data.zero_()
    input = torch.ones((2,4))
    input[0,:].zero_() # input first row is all zero, second row is all one
    target = 4 * torch.ones((2,))
    c_model = nn.Sequential(torch.nn.Linear(4, 2))
    s_model = nn.Sequential(torch.nn.Linear(2, 1))

    c_model.apply(test_init_weights) # initialize weights to 1
    s_model.apply(test_init_weights) # initialize weights to 1, so the expected output will be 8

    criterion = nn.MSELoss()

    s_optimier = torch.optim.Adam(list(s_model.parameters()), lr=1e-3)
    c_optimier = torch.optim.Adam(list(c_model.parameters()), lr=1e-3)

    steps = 2000
    s_train_instance = create_sflserver_instance(s_model, criterion)
    c_train_instance = create_sflclient_instance(c_model)

    c_train_instance.train()
    s_train_instance.train()
    for s in range(steps):
        
        s_optimier.zero_grad()
        c_optimier.zero_grad()
        #client forward:
        hidden = c_train_instance.forward(input)

        #server_compute
        error, ext_grad = s_train_instance.compute(hidden, target)

        if s == steps - 1 or s % 50 == 0:
            print(f"step {s}, error {error}")

        #client backward:
        c_train_instance.backward(ext_grad)

        s_optimier.step()
        c_optimier.step()
    