'''
FL basic functionality, wrap in a class. 

Can be extended to real-applcations if communication protocol is considered.

Refer to McMahan et al. Communication-efficient learning of deep networks from decentralized data for technical details.

'''

import torch
from functions.base_funtions import base_simulator, create_base_instance

class fl_simulator(base_simulator):
    def __init__(self, model, criterion, train_loader, test_loader, args) -> None:
        super().__init__(model, criterion, train_loader, test_loader, args)
        if self.model.get_num_of_cloud_layer() != 0:
            self.model.resplit(0)

        # Create instances
        self.c_instance_list = []
        for i in range(args.num_client):
            self.c_instance_list.append(create_client_instance(self.model.local_list[i]))
        
        # Set optimizer
        self.c_optimizer_list = [None for i in range(args.num_client)]
        for i in range(args.num_client):
            self.c_optimizer_list[i] = torch.optim.SGD(list(self.c_instance_list[i].model.parameters()), lr=args.c_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # Set optimizer scheduler
        milestones = [60, 120, 160]
        self.c_scheduler_list = [None for i in range(args.num_client)]
        for i in range(args.num_client):
            self.c_scheduler_list[i] = torch.optim.lr_scheduler.MultiStepLR(self.c_optimizer_list[i], milestones=milestones, gamma=0.2)  # learning rate decay
 
class create_client_instance(create_base_instance):
    def __init__(self, model) -> None:
        super().__init__(model)
    
    def forward(self, input):
        return self.model(input)
    
    def __call__(self, input):
        return self.forward(input)
