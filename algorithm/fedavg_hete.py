from .fedbase import BasicServer, BasicClient
import numpy as np 

class Server(BasicServer):
    def __init__(self, option, model, clients, valid_data=None, test_data = None):
        super(Server, self).__init__(option, model, clients, valid_data, test_data)
        self.paras_name = ['sample_weights', 'model_type']
        self.model_type = option['model_type']
        
    def sample(self):
        """Sample the clients.
        :param
            replacement: sample with replacement or not
        :return
            a list of the ids of the selected clients
        """
        all_clients = [cid for cid in range(self.num_clients)]
        selected_clients = []
        # collect all the active clients at this round and wait for at least one client is active and
        active_clients = []
        while(len(active_clients)<1):
            active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active() and self.clients[cid].model_type==self.model_type]

        # sample clients
        if self.sample_option == 'active':
            # select all the active clients without sampling
            selected_clients = active_clients
        if self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
        # drop the selected but inactive clients
        selected_clients = list(set(active_clients).intersection(selected_clients))

        return selected_clients

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.sample_weights = np.array(option['sample_weights'])/sum(option['sample_weights'])
        self.temp = option['temp']
        self.model_type = np.random.choice(3, p=self.sample_weights)