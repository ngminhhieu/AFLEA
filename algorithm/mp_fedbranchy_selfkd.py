from cmath import isnan
from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch.nn as nn
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
            losses = [0,0]
            eval_metrics = [0,0,0]
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                tdata = self.calculator.data_to_device(batch_data, device)
                output1, output0 = model(tdata[0], 1)

                for i, output in enumerate([output0, output1]):
                    bmean_loss = self.calculator.lossfunc(output, tdata[-1])
                    losses[i] += bmean_loss.item() * len(batch_data[1])

                    y_pred = output.data.max(1, keepdim=True)[1]
                    correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
                    bmean_eval_metric = (1.0 * correct / len(tdata[1])).item()
                    eval_metrics[i] += bmean_eval_metric * len(batch_data[1])

                ensemble_output = (output0+output1)/2
                ensemble_y_pred = ensemble_output.data.max(1, keepdim=True)[1]
                ensemble_correct = ensemble_y_pred.eq(tdata[1].data.view_as(ensemble_y_pred)).long().cpu().sum()
                ensemble_bmean_eval_metric = (1.0 * ensemble_correct / len(tdata[1])).item()
                eval_metrics[2] += ensemble_bmean_eval_metric * len(batch_data[1])

            for i in range(2):
                eval_metrics[i] /= len(self.test_data)
                losses[i] /= len(self.test_data)
            eval_metrics[2] /= len(self.test_data)

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
            # branches = [int(i) for i in key.split('_')[0]]
            if 'branch_{}'.format(model_types[0]) in key:
                w = weights[0]
                w_avg[key] *= w
            else:
                w = 0
                w_avg[key] = 0
                
            for i in range(1, len(state_dicts)):
                if model_types[i] in key:
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
        self.self_kd = option['selfkd']
        self.model_type = 0 if np.random.rand() < option['small_machine_rate'] else 1


    def test(self, model, dataflag='valid', device=torch.device('cuda')):
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
        ensemble_eval_metric = 0
        
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            with torch.no_grad():
                tdata = self.calculator.data_to_device(batch_data, device)
                outputs, output2 = model(tdata[0], self.model_type)

                bmean_loss = self.calculator.lossfunc(outputs, tdata[-1])
                loss += bmean_loss.item() * len(batch_data[1])

                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
                bmean_eval_metric = (1.0 * correct / len(tdata[1])).item()
                eval_metric += bmean_eval_metric * len(batch_data[1])

                if output2 is not None:
                    ensemble_output = (outputs+output2)/2
                    ensemble_y_pred = ensemble_output.data.max(1, keepdim=True)[1]
                    ensemble_correct = ensemble_y_pred.eq(tdata[1].data.view_as(ensemble_y_pred)).long().cpu().sum()
                    ensemble_bmean_eval_metric = (1.0 * ensemble_correct / len(tdata[1])).item()
                    ensemble_eval_metric += ensemble_bmean_eval_metric * len(batch_data[1])
                else:
                    ensemble_eval_metric += bmean_eval_metric * len(batch_data[1])

        eval_metric =1.0 * eval_metric / len(dataset)
        ensemble_eval_metric =1.0 * ensemble_eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        
        return (eval_metric, ensemble_eval_metric), loss

    def data_to_device(self, data, device=torch.device('cuda')):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)

    def train(self, model, device):
        model = model.to(device)
        model.train()
        
        if self.kd_factor >0:
            src_model = copy.deepcopy(model).to(device)
            src_model.freeze_grad()
        else:
            src_model = None 

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
        outputs_s, representations_s  = model.pred_and_rep(tdata[0], self.model_type)                  # Student
        # outputs_t , _ = src_model.pred_and_rep(tdata[0], self.model_type)                    # Teacher

        kl_loss = 0
        if src_model is not None:
            outputs_t , representations_t = src_model.pred_and_rep(tdata[0], self.model_type)     
            kl_loss += sum(KL_divergence(rt, rs, device) for rt, rs in zip(representations_t, representations_t))
        if self.self_kd:
            for i, representation_s in enumerate(representations_s):
                if i!=len(representations_s)-1:
                    kl_loss += KL_divergence(representations_s[-1].detach(), representation_s, device)
        if type(outputs_s) ==list:
            loss = sum([self.lossfunc(output_s, tdata[1]) for output_s in outputs_s])
        else:
            loss = self.lossfunc(outputs_s, tdata[1])
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
