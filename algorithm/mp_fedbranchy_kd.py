from cmath import isnan
from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import os
import copy

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        
    def finish(self, model_path):
        if not Path(model_path).exists():
            os.system(f"mkdir -p {model_path}")
        task = self.option['task']
        torch.save(self.model.state_dict(), f"{model_path}/{self.name}_{self.num_rounds}_{task}.pth")
        pass
    

    def iterate(self, t, pool):
        self.selected_clients = self.sample()
        models, train_losses, model_types = self.communicate(self.selected_clients,pool)
        if not self.selected_clients: 
            return
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        # self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])

        state_dict = self.average_weights(models, model_types, [self.client_vols[cid] for cid in self.selected_clients])
        self.model.load_state_dict(state_dict)
        return

    def test(self, model=None, device=None):
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
            losses = [0,0]
            eval_metrics = [0,0]
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                for i in range(2):
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data, device, i)
                    losses[i] += bmean_loss * len(batch_data[1])
                    eval_metrics[i] += bmean_eval_metric * len(batch_data[1])
            for i in range(2):
                eval_metrics[i] /= len(self.test_data)
                losses[i] /= len(self.test_data)
            return eval_metrics, losses
        else: 
            return -1, -1
            
    def average_weights(self, models, model_types, weights):
        """
        Returns the average of the weights.
        """
        state_dicts = [model.state_dict() for model in models]
        w_avg = copy.deepcopy(state_dicts[0])
        for key in w_avg.keys():
            if key.startswith('base'):
                w = weights[0]
                w_avg[key] *= w
                for i in range(1, len(state_dicts)):
                    w_avg[key] += weights[i]*state_dicts[i][key]
                    w += weights[i]

                w_avg[key] = w_avg[key]/ w
            elif key.startswith('branch1'):
                if model_types[0] == 0:
                    w = weights[0]
                    w_avg[key] *= w
                else:
                    w = 0
                    w_avg[key] = 0
                for i in range(1, len(state_dicts)):
                    if model_types[i] == 0:
                        w_avg[key] += weights[i] * state_dicts[i][key]
                        w += weights[i]
                if w > 0:
                    w_avg[key] = w_avg[key]/ w
                else:
                    w_avg[key] = state_dicts[0][key]
            elif key.startswith('branch2'):
                if model_types[0] == 1:
                    w = weights[0]
                    w_avg[key] *= w
                else:
                    w = 0
                    w_avg[key] = 0
                for i in range(1, len(state_dicts)):
                    if model_types[i] == 1:
                        w_avg[key] += weights[i] * state_dicts[i][key]
                        w += weights[i]
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

class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.kd_factor = option['mu']
        self.T = 10        
        self.model_type = np.random.randint(0,2)


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
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data,device, self.model_type)

            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])

        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        
        return eval_metric, loss

    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        src_model = copy.deepcopy(model).to(device)
        src_model.freeze_grad()
                
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        # if self.model_type==0:
        #     optimizer = self.calculator.get_optimizer(self.optimizer_name, model.branch1(), lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        # else:
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, kl_loss = self.get_loss(model, src_model, batch_data, device)
                loss = loss + self.kd_factor * kl_loss
                loss.backward()
                optimizer.step()
        return
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, src_model, data, device):
        tdata = self.data_to_device(data, device)
        output_s, _ = model.pred_and_rep(tdata[0], self.model_type)                  # Student

        if self.kd_factor >0:
            output_t , _ = src_model.pred_and_rep(tdata[0], self.model_type)                    # Teacher
            kl_loss = nn.KLDivLoss()(F.log_softmax(output_s/self.T, dim=1),
                                F.softmax(output_t/self.T, dim=1))    # KL divergence
        else:
            kl_loss = 0

        loss = self.lossfunc(output_s, tdata[1])
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
