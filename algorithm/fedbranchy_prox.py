from cmath import isnan
from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from .fedbase import BasicServer, BasicClient
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import torch.nn.functional as F
import torch
import os
import copy


def KL_divergence(teacher_batch_input, student_batch_input, device):
    """
    Compute the KL divergence of 2 batches of layers
    Args:
        teacher_batch_input: Size N x d
        student_batch_input: Size N x c
    
    Method: Kernel Density Estimation (KDE)
    Kernel: Gaussian
    Author: Nguyen Nang Hung
    """
    batch_student, _ = student_batch_input.shape
    batch_teacher, _ = teacher_batch_input.shape
    
    assert batch_teacher == batch_student, "Unmatched batch size"
    
    sub_s_norm = torch.cdist(student_batch_input,student_batch_input).flatten()[1:].view(batch_student-1, batch_student+1)[:,:-1].reshape(batch_student, batch_student-1)
    std_s = torch.std(sub_s_norm)
    mean_s = torch.mean(sub_s_norm)
    kernel_mtx_s = torch.pow(sub_s_norm - mean_s, 2) / (torch.pow(std_s, 2) + 0.001)
    kernel_mtx_s = torch.exp(-1/2 * kernel_mtx_s)
    kernel_mtx_s = kernel_mtx_s/torch.sum(kernel_mtx_s, dim=1, keepdim=True)
    
    sub_t_norm = torch.cdist(teacher_batch_input,teacher_batch_input).flatten()[1:].view(batch_teacher-1, batch_teacher+1)[:,:-1].reshape(batch_teacher, batch_teacher-1)
    std_t = torch.std(sub_t_norm)
    mean_t = torch.mean(sub_t_norm)
    kernel_mtx_t = torch.pow(sub_t_norm - mean_t, 2) / (torch.pow(std_t, 2) + 0.001)
    kernel_mtx_t = torch.exp(-1/2 * kernel_mtx_t)
    kernel_mtx_t = kernel_mtx_t/torch.sum(kernel_mtx_t, dim=1, keepdim=True)
    
    kl = torch.sum(kernel_mtx_t * torch.log(kernel_mtx_t/kernel_mtx_s))
    return kl


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_branches = 2

    def finish(self, model_path):
        if not Path(model_path).exists():
            os.system(f"mkdir -p {model_path}")
        task = self.option['task']
        torch.save(self.model.state_dict(), f"{model_path}/{self.name}_{self.num_rounds}_{task}.pth")
        pass
    

    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses, model_types = self.communicate(self.selected_clients)
        from collections import Counter
        print(Counter(model_types))
        if not self.selected_clients: 
            return
        device0 = torch.device(f"cuda")
        models = [i.to(device0) for i in models]
        self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])

        # state_dict = self.average_weights(models, model_types, [self.client_vols[cid] for cid in self.selected_clients])
        # self.model.load_state_dict(state_dict)
        return

    def test(self, model=None, device=torch.device('cuda')):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model==None: 
            model=self.model
        if self.test_data:
            model.eval()
            losses = 0
            eval_metrics =0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data, device)
                losses += bmean_loss * len(batch_data[1])
                eval_metrics += bmean_eval_metric * len(batch_data[1])
            eval_metrics /= len(self.test_data)
            losses /= len(self.test_data)
            return eval_metrics, losses
        else: 
            return -1, -1
            
    def average_weights(self, models, model_types, weights):
        """
        Returns the average of the weights.
        """
        factors = {0:5, 1:2, 2:1}
        state_dicts = [model.state_dict() for model in models]
        w_avg = copy.deepcopy(state_dicts[0])
        for key in w_avg.keys():
            branches = [int(i) for i in key.split('_')[0][1:]]
            if model_types[0] in branches:
                w = factors[model_types[0]]*weight_norm(weights[0], dim=None)
                w_avg[key] *= weight_norm(weights[0], dim=None)*factors[model_types[0]]
            else:
                w = 0
                w_avg[key] = 0
                
            for i in range(1, len(state_dicts)):
                if model_types[i] in branches:
                    w_avg[key] += factors[model_types[i]]*weight_norm(weights[i], dim=None) * state_dicts[i][key]
                    w += weight_norm(weights[i], dim=None)*factors[model_types[i]]
            if w > 0:
                w_avg[key] = w_avg[key]/ w
            else:
                w_avg[key] = state_dicts[0][key]
               
        return w_avg

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        model_types = [cp["model_type"] for cp in packages_received_from_clients]
        return models, train_losses, model_types

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        if 'loss_weight' in option:
            self.lossfunc = nn.CrossEntropyLoss(torch.tensor(option['loss_weight']).cuda())
        else:
            self.lossfunc = nn.CrossEntropyLoss()
        self.kd_factor = option['mu']
        self.self_kd = option['selfkd']
        self.weighted = option['weighted']
        self.T = 3
        self.model_type = np.random.randint(0,self.T)
        self.step=0
    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model,torch.device('cuda'))
        cpkg = self.pack(model, loss)
        return cpkg

    def test(self, model, dataflag='valid', device='cpu'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model = model.to(device)
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data,device)

            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])

        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        
        return eval_metric, loss

    def train(self, model, device):
        model = model.to(device)
        model.train()
    
        if self.kd_factor >0:
            src_model = copy.deepcopy(model).to(device)
            src_model.freeze_grad()
        else:
            src_model = None 
            
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss,kl_loss = self.get_loss(model, batch_data, device, src_model)
                loss = loss + self.kd_factor * kl_loss
                loss.backward()
                optimizer.step()
        return
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, data, device, src_model=None):
        tdata = self.data_to_device(data, device)    
        outputs_s, representations_s  = model.pred_and_rep(tdata[0], self.model_type)                  # Student
        if type(outputs_s) == list:
            loss = sum([self.lossfunc(output_s, tdata[1]) for output_s in outputs_s])
        else:
            loss = self.lossfunc(outputs_s, tdata[1])
        kl_loss = 0
        if self.kd_factor > 0:
            for pm, ps in zip(model.parameters(), src_model.parameters()):
                kl_loss += torch.sum(torch.pow(pm-ps,2))

        return loss, kl_loss

    def pack(self, model, loss):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "train_loss": loss,
            "model_type": self.model_type
        }
