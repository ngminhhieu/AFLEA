from cmath import isnan
from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
from .fedbase import BasicServer, BasicClient
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import os
import copy
from collections import defaultdict

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.global_protos = {}
        
    
    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "global_protos" : self.global_protos,
            "model" : copy.deepcopy(self.model),
        }

    def iterate(self, t):
        self.selected_clients = self.sample()
        local_protos, train_losses = self.communicate(self.selected_clients)
        if not self.selected_clients: 
            return
        self.global_protos = self.proto_aggregation(local_protos)
        return

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for local_protos in local_protos_list:
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label
        
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
            losses = []
            eval_metrics = []
            for c in self.clients:
                loss, eval_metric = 0, 0
                if c.state_dict is not None:
                    model.load_state_dict(c.state_dict)
                model.eval()
                data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
                for batch_id, batch_data in enumerate(data_loader):
                    bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data, device, c.model_type)
                    loss += bmean_loss * len(batch_data[1])
                    eval_metric += bmean_eval_metric * len(batch_data[1])
                eval_metric /= len(self.test_data)
                loss /= len(self.test_data)
                losses.append(loss)
                eval_metrics.append(eval_metric)
            return (np.mean(eval_metrics), np.min(eval_metrics), np.max(eval_metrics)), np.mean(losses)
        else: 
            return -1, -1

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        local_protos = [cp["local_protos"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        return local_protos, train_losses

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.model_type = np.random.randint(0,2)
        self.state_dict = None 
        self.local_protos= defaultdict(list)    
        self.global_protos = {}
        self.kd_factor = 1

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model'], received_pkg['global_protos']

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
        device =torch.device('cuda')
        model, global_protos = self.unpack(svr_pkg)
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        else:
            self.state_dict = copy.deepcopy(model.state_dict())
        loss = self.train_loss(model)
        self.train(model, global_protos, device)
        cpkg = self.pack(loss)
        return cpkg

    def pack(self, loss):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        local_protos = {k: torch.stack(v).mean(0) for k,v in self.local_protos.items()}
        return {
            "local_protos" : local_protos,
            "train_loss": loss,
        }

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
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        else:
            self.state_dict = copy.deepcopy(model.state_dict())
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

    def train(self, model, global_protos, device):
        model = model.to(device)
        model.train()
        self.local_protos= defaultdict(list)    
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss, proto_loss  = self.get_loss(model, global_protos, batch_data, device)
                loss = loss + self.kd_factor * proto_loss
                loss.backward()
                optimizer.step()
        self.state_dict = copy.deepcopy(model.state_dict())

        return
    
    
    def data_to_device(self, data,device):
        return data[0].to(device), data[1].to(device)


    def get_loss(self, model, global_protos, data, device):
        tdata = self.data_to_device(data, device)
        output_s, protos = model.pred_and_rep(tdata[0], self.model_type)                  # Student

        
        if self.kd_factor >0 and len(global_protos) >0:
            # print(global_protos)
            loss_mse = nn.MSELoss()
            proto_new = copy.deepcopy(protos.data)

        for i, label in enumerate(tdata[1]):
            if self.kd_factor >0 and len(global_protos) >0:
                if label.item() in global_protos.keys():
                    proto_new[i, :] = global_protos[label.item()][0].data
                
            self.local_protos[label.item()].append(protos[i,:].detach())

        if self.kd_factor >0 and len(global_protos) >0:
            kl_loss = loss_mse(proto_new, protos)
        else:
            kl_loss = 0

        loss = self.lossfunc(output_s, tdata[1])

        return loss, kl_loss

