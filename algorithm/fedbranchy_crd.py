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

from .crd_utils import CRDLoss


class Server(BasicServer):
    def __init__(self, option, model, clients, valid_data = None, test_data = None, **kwargs):
        super(Server, self).__init__(option, model, clients, valid_data, test_data, **kwargs)
        self.paras_name = ['sample_weights', 'agg_weights', 'nce_t']
        self.factors = {i:j for i, j in enumerate(option['agg_weights'])}

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
        # self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])

        state_dict = self.average_weights(models, model_types, [self.client_vols[cid] for cid in self.selected_clients])
        self.model.load_state_dict(state_dict)
        return

    def average_weights(self, models, model_types, weights):
        """
        Returns the average of the weights.
        """

        state_dicts = [model.state_dict() for model in models]
        w_avg = copy.deepcopy(state_dicts[0])
        for key in w_avg.keys():
            branches = [int(i) for i in key.split('_')[0][1:]]
            if model_types[0] in branches:
                w = self.factors[model_types[0]]*weights[0]
                w_avg[key] *= weights[0]*self.factors[model_types[0]]
            else:
                w = 0
                w_avg[key] = 0
                
            for i in range(1, len(state_dicts)):
                if model_types[i] in branches:
#                     print(key, model_types[i], torch.abs(state_dicts[i][key]*1.0).mean(), torch.abs(state_dicts[i][key]*1.0).std())
                    w_avg[key] += self.factors[model_types[i]]*weights[i] * state_dicts[i][key]
                    w += weights[i]*self.factors[model_types[i]]
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
            self.lossfunc = nn.CrossEntropyLoss().cuda()

        if option['model'] == 'efficientnet':
            feat_dim = 128 
        elif option['model'] == 'inception':
            feat_dim = 256 
        else:
            feat_dim = 4 * option['base_dim']
        self.crdloss = CRDLoss(n_data = len(train_data), s_dim = feat_dim, t_dim = feat_dim, nce_t=option['nce_t']).cuda()
        self.kd_factor = option['mu']
        self.sample_weights = np.array(option['sample_weights'])/sum(option['sample_weights'])
        self.model_type = np.random.choice(len(option['sample_weights']), p=self.sample_weights)
        self.step = 0

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

        trainable_list = nn.ModuleList([model, self.crdloss.embed_s, self.crdloss.embed_t])

        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, droplast=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, trainable_list, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        # need to add parameter of loss 
        
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss,kl_loss = self.get_loss(model, batch_data, device, src_model)
#                 print(loss, kl_loss)
                loss = loss + self.kd_factor * kl_loss
                loss.backward()
                optimizer.step()
        return
    
    
    def data_to_device(self, data,device):
        return [d.to(device) for d in data]


    def get_loss(self, model, data, device, src_model=None):
        data = self.data_to_device(data, device)   
        if len(data) == 4:
            img, target, index, contrast_idx = data
        else:
            img, target = data
            index, contrast_idx = None, None 

        outputs_s, representations_s  = model.pred_and_rep(img, self.model_type)                  # Student
        if type(outputs_s) == list:
            loss = sum([w*self.lossfunc(output_s, target) for w,output_s in zip([10,5,1], outputs_s)])/sum([10,5,1][:len(outputs_s)])
        else:
            loss = self.lossfunc(outputs_s, target)
        kl_loss = 0
        if self.kd_factor > 0 and index is not None and contrast_idx is not None:

            outputs_t , representations_t = src_model.pred_and_rep(img, self.model_type)     
            # if self.temp <= 0:
            #     kl_loss = sum(KL_divergence(rt, rs, device) for rt, rs in zip(representations_t, representations_s))
            # elif type(outputs_s) == list:
            #     kl_loss = self.temp**2 * nn.KLDivLoss()(F.log_softmax(outputs_s[-1]/self.temp, dim=1),
            #                         F.softmax(outputs_t[-1]/self.temp, dim=1)) 
            # else:
            #     kl_loss = self.temp**2 * nn.KLDivLoss()(F.log_softmax(outputs_s/self.temp, dim=1),
            #                             F.softmax(outputs_t/self.temp, dim=1)) 
            # if index is not None and contrast_idx is not None:
            
            kl_loss += self.crdloss(representations_s[-1], representations_t[-1], index, contrast_idx)

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
