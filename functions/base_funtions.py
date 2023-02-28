'''
SFL basic functionality, wrap in a class. 

Can be extended to real-applcations if communication protocol is considered.

To understand how the training works in this implementation of SFL. We provide a tutorial in __main__ function:

Refer to Thapa et al. https://arxiv.org/abs/2004.12088 for technical details.

'''

from email.policy import strict
import torch
import logging
from utils import AverageMeter, accuracy, average_weights
from utils import setup_logger
class base_simulator:
    def __init__(self, model, criterion, train_loader, test_loader, args) -> None:
        if not model.cloud_classifier_merge:
            model.merge_classifier_cloud()
        model_log_file = args.output_dir + '/output.log'
        self.logger = setup_logger('default_logger', model_log_file, level=logging.DEBUG)
        self.model = model
        self.criterion = criterion
        self.num_client = args.num_client
        self.num_epoch = args.num_epoch
        self.num_class = args.num_class
        self.batch_size = args.batch_size
        self.output_dir = args.output_dir
        self.div_lambda = [args.div_lambda for _ in range(args.num_client)]
        self.auto_scaler = True
        self.client_sample_ratio  = args.client_sample_ratio

        # set dummy variables
        self.s_instance = None
        self.c_instance_list = []
        self.s_optimizer = None
        self.c_optimizer_list = []
        self.s_scheduler = None
        self.c_scheduler_list = []

        #initialize data iterator
        self.client_dataloader = train_loader
        self.validate_loader = test_loader
        self.client_iterator_list = []
        if train_loader is not None:
            for client_id in range(args.num_client):
                # train_loader[client_id].persistent_workers = True #TODO: remove if hurts
                self.client_iterator_list.append(create_iterator(iter((train_loader[client_id]))))
    
    def next_data_batch(self, client_id):
        try:
            images, labels = next(self.client_iterator_list[client_id])
            if images.size(0) != self.batch_size:
                try:
                    next(self.client_iterator_list[client_id])
                except StopIteration:
                    pass
                self.client_iterator_list[client_id] = create_iterator(iter((self.client_dataloader[client_id])))
                images, labels = next(self.client_iterator_list[client_id])
        except StopIteration:
            self.client_iterator_list[client_id] = create_iterator(iter((self.client_dataloader[client_id])))
            images, labels = next(self.client_iterator_list[client_id])
        return images, labels

    def optimizer_zero_grads(self):  # This needs to be called
        if self.s_optimizer is not None:
            self.s_optimizer.zero_grad()
        if self.c_optimizer_list: 
            for i in range(self.num_client):
                self.c_optimizer_list[i].zero_grad()

    def fedavg(self, pool = None, divergence_aware = False, divergence_measure = False):
        
        global_weights = average_weights(self.model.local_list, pool)
        if divergence_measure:
            divergence_list = []
        for i in range(self.num_client):
            
            if divergence_measure:
                if pool is None:
                    pool = range(len(self.num_client))
                
                if i in pool: # if current client is selected.
                    weight_divergence = 0.0
                    for key in global_weights.keys():
                        if "running" in key or "num_batches" in key: # skipping batchnorm running stats
                            continue
                        weight_divergence += torch.linalg.norm(torch.flatten(self.model.local_list[i].state_dict()[key] - global_weights[key]).float(), dim = -1, ord = 2)
                    divergence_list.append(weight_divergence.item())

            if divergence_aware:
                '''
                [1]DAPU: Zhuang et al. - Collaborative Unsupervised Visual Representation Learning from Decentralized Data
                [2]Divergence_aware Zhuang et al. - Divergence-aware Federated Self-Supervised Learning

                Difference: [1] it is only used for the MLP predictor part in the online encoder (the fast one) in BYOL, not in any of the backbone model.
                            [2] it is used for the entire online encoder as well as its predictor. auto_scaler is invented.
                            
                '''
                if pool is None:
                    pool = range(len(self.num_client))
                
                if i in pool: # if current client is selected.
                    weight_divergence = 0.0
                    for key in global_weights.keys():
                        if "running" in key or "num_batches" in key: # skipping batchnorm running stats
                            continue
                        weight_divergence += torch.linalg.norm(torch.flatten(self.model.local_list[i].state_dict()[key] - global_weights[key]).float(), dim = -1, ord = 2)
                    
                    mu = self.div_lambda[i] * weight_divergence.item() # the choice of dic_lambda depends on num_param in client-side model
                    mu = 1 if mu >= 1 else mu # If divergence is too large, just do personalization & don't consider the average.

                    for key in global_weights.keys():
                        self.model.local_list[i].state_dict()[key] = mu * self.model.local_list[i].state_dict()[key] + (1 - mu) * global_weights[key]

                    if self.auto_scaler: # is only done at epoch 1
                        self.div_lambda[i] = mu / weight_divergence # such that next div_lambda will be similar to 1. will not be a crazy value.
                        self.auto_scaler = False # Will only use it once at the first round.
                else: # if current client is not selected.
                    self.model.local_list[i].load_state_dict(global_weights)
            else:
                '''
                Normal case: directly get the averaged result
                '''
                self.model.local_list[i].load_state_dict(global_weights)

        if divergence_measure:
            return divergence_list
        else:
            return None
    def train(self):
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].train()
        if self.s_instance is not None:
            self.s_instance.train()

    def eval(self):
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].eval()
        if self.s_instance is not None:
            self.s_instance.eval()
    
    def cuda(self):
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].cuda()
        if self.s_instance is not None:
            self.s_instance.cuda()

    def cpu(self):
        if self.c_instance_list: 
            for i in range(self.num_client):
                self.c_instance_list[i].cpu()
        if self.s_instance is not None:
            self.s_instance.cpu()

    def validate(self): # validate in cuda mode
        """
        Run evaluation
        """
        top1 = AverageMeter()
        self.eval()  #set to eval mode
        if self.c_instance_list:
            self.c_instance_list[0].cuda()
        if self.s_instance is not None:
            self.s_instance.cuda()
        
        if self.c_instance_list:
            for input, target in self.validate_loader:
                input = input.cuda()
                target = target.cuda()
                with torch.no_grad():
                    output = self.c_instance_list[0](input)
                    if self.s_instance is not None:
                        output = self.s_instance(output)
                prec1 = accuracy(output.data, target)[0]
                top1.update(prec1.item(), input.size(0))

        self.train() #set back to train mode
        return top1.avg

    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"
        torch.save(self.model.cloud.state_dict(), self.output_dir + '/checkpoint_s_{}.tar'.format(epoch))
        torch.save(self.model.local_list[0].state_dict(), self.output_dir + '/checkpoint_c_{}.tar'.format(epoch))
    
    def load_model(self, is_best=True, epoch=200):
        if is_best:
            epoch = "best"
        checkpoint_s = torch.load(self.output_dir + '/checkpoint_s_{}.tar'.format(epoch))
        self.model.cloud.load_state_dict(checkpoint_s)
        checkpoint_c = torch.load(self.output_dir + '/checkpoint_c_{}.tar'.format(epoch))
        for i in range(self.num_client):
            self.model.local_list[i].load_state_dict(checkpoint_c)

    def load_model_from_path(self, model_path, load_client = True, load_server = False):
        if load_server:
            checkpoint_s = torch.load(model_path + '/checkpoint_s_best.tar')
            self.model.cloud.load_state_dict(checkpoint_s)
        if load_client:
            checkpoint_c = torch.load(model_path + '/checkpoint_c_best.tar')
            for i in range(self.num_client):
                self.model.local_list[i].load_state_dict(checkpoint_c)

    def log(self, message):
        self.logger.debug(message)

class create_iterator():
    def __init__(self, iterator) -> None:
        self.iterator = iterator

    def __next__(self):
        return next(self.iterator)

class create_base_instance:
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self):
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def cuda(self):
        self.model.cuda()
    
    def cpu(self):
        self.model.cpu()