from pathlib import Path
from .mp_fedbase import MPBasicServer, MPBasicClient
import torch
import numpy as np 
import os
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn import metrics
import copy 

class Server(MPBasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.num_groups = option['num_groups']
        self.seed = option['seed']

    def compare_model(self, models):
        n_selected_clients = len(self.selected_clients)
        labels_to_clients = defaultdict(list)

        X = []
        for i, (client, model) in enumerate(zip(self.selected_clients, models)):
            labels_to_clients[tuple(sorted(self.clients[client].all_labels))].append(i)
            X.append(model.fc1.weight.data.flatten().detach().cpu().numpy())
        p = np.zeros(len(X))
        y = np.zeros(len(X))
        for i, client_ids in enumerate(labels_to_clients.values()):
            y[client_ids] = i 
            p[client_ids] = 1/len(client_ids)
        kmeans = KMeans(n_clusters=self.num_groups, random_state=self.seed).fit(X)   
        y_pred = kmeans.labels_
        print(metrics.homogeneity_score(y, y_pred), metrics.completeness_score(y, y_pred))
        return p.tolist()

    def aggregate_(self, models):
        """
        Returns the average of the weights.
        """
        state_dicts = [model.state_dict() for model in models]
        K = models[0].fc2.weight.data.shape[0]
        device = next(models[0].parameters()).device

        masks = []
        for client in self.selected_clients:
            mask = np.zeros(K)
            mask[self.clients[client].all_labels] = 1
            masks.append(torch.tensor(mask).to(device))

        avg_state_dict = copy.deepcopy(state_dicts[0])
        for key in avg_state_dict.keys():
            if key=='fc2.weight':
                avg_state_dict[key] = avg_state_dict[key] * masks[0].reshape((-1,1))
                mask = masks[0].reshape((-1,1))
                for i in range(1, len(state_dicts)):
                    avg_state_dict[key] += state_dicts[i][key]*masks[i].reshape((-1,1))
                    mask += masks[i].reshape((-1,1))

                avg_state_dict[key] = avg_state_dict[key]/(mask+10e-10)
            elif key=='fc2.bias':
                avg_state_dict[key] = avg_state_dict[key] * masks[0]
                mask = masks[0]
                for i in range(1, len(state_dicts)):
                    avg_state_dict[key] += state_dicts[i][key]*masks[i]
                    mask += masks[i]

                avg_state_dict[key] = avg_state_dict[key]/(mask+10e-10)
            else:
                for i in range(1, len(state_dicts)):
                    avg_state_dict[key] += state_dicts[i][key]
                avg_state_dict[key] = torch.div(avg_state_dict[key], len(state_dicts))
        self.model.load_state_dict(avg_state_dict)


    def iterate(self, t, pool):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients, pool)
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        p = self.compare_model(models)
        # self.model = self.aggregate(models, p = [1.0 for cid in self.selected_clients])
        self.model = self.aggregate(models, p = p)
        # self.aggregate(models)
        
        return


class Client(MPBasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.all_labels = self.train_data.all_labels if self.train_data else [] 
