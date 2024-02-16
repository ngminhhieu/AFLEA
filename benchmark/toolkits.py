"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
    iid:            0 : identical and independent distributions of the dataset among clients
    label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
                    2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
                    3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
-----------------------------------------------------------------------------------
depends on partitions:
    feature skew:   4 Noise: each party owns data samples of a fixed number of labels.
                    5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
    iid:            6 Vol: only the vol of local dataset varies.
    niid:           7 Vol: for generating synthetic data
-----------------------------------------------------------------------------------
                    8: each groups of client contain only subset of labels, and labels of groups are disjoint
"""
import torch
import ujson
import numpy as np
import os.path
import random
import urllib
import zipfile
import os
import ssl
from torch.utils.data import Dataset, DataLoader
import torch
ssl._create_default_https_context = ssl._create_unverified_context
import importlib
import pickle 
import collections

def set_random_seed(seed=0):
    """Set random seed"""
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url: urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class BasicTaskGen:
    _TYPE_DIST = {
        0: 'iid',
        1: 'label_skew_quantity',
        2: 'label_skew_dirichlet',
        3: 'label_skew_shard',
        4: 'feature_skew_noise',
        5: 'feature_skew_id',
        6: 'iid_volumn_skew',
        7: 'niid_volumn_skew',
        8: 'concept skew',
        9: 'concept and feature skew and balance',
        10: 'concept and feature skew and imbalance',
    }
    _TYPE_DATASET = ['2DImage', '3DImage', 'Text', 'Sequential', 'Graph', 'Tabular']

    def __init__(self, benchmark, dist_id, skewness, rawdata_path, seed=0):
        self.benchmark = benchmark
        self.rootpath = './fedtask'
        self.rawdata_path = rawdata_path
        if not os.path.isdir(self.rawdata_path):
            os.makedirs(self.rawdata_path)
        self.dist_id = dist_id
        self.dist_name = self._TYPE_DIST[dist_id]
        self.skewness = 0 if dist_id==0 else skewness
        self.num_clients = -1
        self.seed = seed
        set_random_seed(self.seed)

    def run(self):
        """The whole process to generate federated task. """
        pass

    def load_data(self):
        """Download and load dataset into memory."""
        pass

    def partition(self):
        """Partition the data according to 'dist' and 'skewness'"""
        pass

    def save_data(self):
        """Save the federated dataset to the task_path/data.
        This algorithm should be implemented as the way to read
        data from disk that is defined by DataReader.read_data()
        """
        pass

    def save_info(self):
        """Save the task infomation to the .json file stored in taskpath"""
        pass

    def get_taskname(self):
        """Create task name and return it."""
        taskname = '_'.join([self.benchmark, 'cnum' +  str(self.num_clients), 'cgroup' +  str(self.num_groups), 'dist' + str(self.dist_id), 'skew' + str(self.skewness).replace(" ", ""), 'seed'+str(self.seed)])
        return taskname

    def get_client_names(self):
        k = str(len(str(self.num_clients)))
        return [('Client{:0>' + k + 'd}').format(i) for i in range(self.num_clients)]

    def create_task_directories(self):
        """Create the directories of the task."""
        taskname = self.get_taskname()
        taskpath = os.path.join(self.rootpath, taskname)
        os.mkdir(taskpath)
        os.mkdir(os.path.join(taskpath, 'record'))

    def _check_task_exist(self):
        """Check whether the task already exists."""
        taskname = self.get_taskname()
        return os.path.exists(os.path.join(self.rootpath, taskname))

class DefaultTaskGen(BasicTaskGen):
    def __init__(self, benchmark, dist_id, skewness, rawdata_path, num_clients=1, num_groups=3, minvol=10, seed=0):
        super(DefaultTaskGen, self).__init__(benchmark, dist_id, skewness, rawdata_path, seed)
        self.minvol=minvol
        self.num_classes = -1
        self.train_data = None
        self.test_data = None
        self.num_clients = num_clients
        self.num_groups = num_groups
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.rootpath, self.taskname)
        self.save_data = self.XYData_to_json
        self.datasrc = {
            'lib': None,
            'class_name': None,
            'args':[]
        }

    def run(self):
        """ Generate federated task"""
        # check if the task exists
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print('-----------------------------------------------------')
        print('Loading...')
        self.load_data()
        print('Done.')
        # partition data and hold-out for each local dataset
        print('-----------------------------------------------------')
        print('Partitioning data...')
        train_cidxs = self.partition()
        # train_cidxs, valid_cidxs = self.local_holdout(local_datas, rate=0.8, shuffle=True)
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        self.save_info()
        self.save_data(train_cidxs)
        print('Done.')
        return

    def load_data(self):
        """ load and pre-process the raw data"""
        return

    def partition(self):
        # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
        if self.dist_id == 0:
            """IID"""
            d_idxs = np.random.permutation(len(self.train_data))
            local_datas = np.array_split(d_idxs, self.num_clients)

        elif self.dist_id == 1:
            """label_skew_quantity"""
            self.skewness = min(max(0, self.skewness),1.0)
            # pair is (id, label)
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            num = max(int(1.0*self.num_classes-self.skewness*self.num_classes), 1) # each client contains only 'num' labels
            K = self.num_classes
            local_datas = [[] for _ in range(self.num_clients)]
            if num == K:
                for k in range(K):
                    # get list of ids which has label k
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for cid in range(self.num_clients):
                        local_datas[cid].extend(split[cid].tolist())
            else:
                times = [0 for _ in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = [i % K] # set of label appear in client i
                    times[i % K] += 1 # the total number of appearance of that label in all client
                    j = 1 # the current size of the label set
                    while (j < num):
                        ind = random.randint(0, K - 1) # get a random label
                        if (ind not in current): # if label not in current label set of the client
                            j = j + 1 
                            current.append(ind) # add that label to the current label set
                            times[ind] += 1 
                    contain.append(current)
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[k])
                    ids = 0
                    # distribute subset of ids w.r.t the label to all clients having that label
                    for cid in range(self.num_clients):
                        if k in contain[cid]:
                            local_datas[cid].extend(split[ids].tolist())
                            ids += 1

        elif self.dist_id == 2:
            """label_skew_dirichlet"""
            min_size = 0
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            local_datas = [[] for _ in range(self.num_clients)]
            while min_size < self.minvol:
                idx_batch = [[] for i in range(self.num_clients)]
                for k in range(self.num_classes):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < len(self.train_data)/ self.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_clients):
                np.random.shuffle(idx_batch[j])
                local_datas[j].extend(idx_batch[j])

        elif self.dist_id == 3:
            """label_skew_shard"""
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            self.skewness = min(max(0, self.skewness), 1.0)
            num_shards = max(int(2*(1.0*self.num_classes-self.skewness*self.num_classes)), 1) # number of shards in each client
            client_datasize = int(len(self.train_data) / self.num_clients) # size of data in each client

            # sorted by label
            all_idxs, labels = zip(*sorted(dpairs, key=lambda x:x[1]))
            # size of each shard
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            local_datas = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                for rand in rand_set:
                    local_datas[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])

        elif self.dist_id == 4:
            MIN_ALPHA = 0.01
            alpha = (-4*np.log(self.skewness + 10e-8))**4
            alpha = max(alpha, MIN_ALPHA)
            labels = [self.train_data[did][-1] for did in range(len(self.train_data))]
            lb_counter = collections.Counter(labels)
            p = np.array([1.0*v/len(self.train_data) for v in lb_counter.values()])
            lb_dict = {}
            labels = np.array(labels)
            for lb in range(len(lb_counter.keys())):
                lb_dict[lb] = np.where(labels==lb)[0]
            proportions = [np.random.dirichlet(alpha*p) for _ in range(self.num_clients)]
            while np.any(np.isnan(proportions)):
                proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
            while True:
                # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
                mean_prop = np.mean(proportions, axis=0)
                error_norm = ((mean_prop-p)**2).sum()
                print("Error: {:.8f}".format(error_norm))
                if error_norm<=1e-2/self.num_classes:
                    break
                exclude_norms = []
                for cid in range(self.num_clients):
                    mean_excid = (mean_prop*self.num_clients-proportions[cid])/(self.num_clients-1)
                    error_excid = ((mean_excid-p)**2).sum()
                    exclude_norms.append(error_excid)
                excid = np.argmin(exclude_norms)
                sup_prop = [np.random.dirichlet(alpha*p) for _ in range(self.num_clients)]
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    mean_alter_cid = mean_prop - proportions[excid]/self.num_clients + sup_prop[cid]/self.num_clients
                    error_alter = ((mean_alter_cid-p)**2).sum()
                    alter_norms.append(error_alter)
                if len(alter_norms)>0:
                    alcid = np.argmin(alter_norms)
                    proportions[excid] = sup_prop[alcid]
            local_datas = [[] for _ in range(self.num_clients)]
            # self.dirichlet_dist = [] # for efficiently visualizing
            for lb in lb_counter.keys():
                lb_idxs = lb_dict[lb]
                lb_proportion = np.array([pi[lb] for pi in proportions])
                lb_proportion = lb_proportion/lb_proportion.sum()
                lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
                lb_datas = np.split(lb_idxs, lb_proportion)
                # self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
                local_datas = [local_data+lb_data.tolist() for local_data,lb_data in zip(local_datas, lb_datas)]
            # self.dirichlet_dist = np.array(self.dirichlet_dist).T
            for i in range(self.num_clients):
                np.random.shuffle(local_datas[i])

        elif self.dist_id == 5:
            """feature_skew_id"""
            if not isinstance(self.train_data, TupleDataset):
                raise RuntimeError("Support for dist_id=5 only after setting the type of self.train_data is TupleDataset")
            Xs, IDs, Ys = self.train_data.tolist()
            self.num_clients = len(set(IDs))
            local_datas = [[] for _ in range(self.num_clients)]
            for did in range(len(IDs)):
                local_datas[IDs[did]].append(did)

        elif self.dist_id == 6:
            minv = 0
            d_idxs = np.random.permutation(len(self.train_data))
            while minv < self.minvol:
                proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * len(self.train_data))
            proportions = (np.cumsum(proportions) * len(d_idxs)).astype(int)[:-1]
            local_datas  = np.split(d_idxs, proportions)

        elif self.dist_id == 8:
            """label_skew_quantity"""
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]

            K = self.num_classes
            labels = np.arange(K)
            np.random.shuffle(labels)
            # each split is a set of labels of a group
            label_splits = np.array_split(labels, self.num_groups)
            
            client_ids = np.arange(self.num_clients)
            np.random.shuffle(client_ids)
            client_ids_splits = np.array_split(client_ids, self.num_groups)

            local_datas = [[] for _ in range(self.num_clients)]
            for labels, clients in zip(label_splits, client_ids_splits):
                for k in labels:
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    id_splits = np.array_split(idx_k, len(clients))
                    for client_id, data_id in zip(clients, id_splits):
                        local_datas[client_id].extend(data_id.tolist())
            
        return local_datas

    def local_holdout(self, local_datas, rate=0.8, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * rate)
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        return train_cidxs, valid_cidxs

    def save_info(self):
        info = {
            'benchmark': self.benchmark,  # name of the dataset
            'dist': self.dist_id,  # type of the partition way
            'skewness': self.skewness,  # hyper-parameter for controlling the degree of niid
            'num-clients': self.num_clients,  # numbers of all the clients
        }
        # save info.json
        with open(os.path.join(self.taskpath, 'info.json'), 'w') as outf:
            ujson.dump(info, outf)

    def convert_data_for_saving(self):
        """Convert self.train_data and self.test_data to list that can be stored as .json file and the converted dataset={'x':[], 'y':[]}"""
        pass

    def XYData_to_json(self, train_cidxs, valid_cidxs=None):
        self.convert_data_for_saving()
        # save federated dataset
        feddata = {
            'store': 'XY',
            'client_names': self.cnames,
            'dtest': self.test_data,
            'dvalid': self.val_data

        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain':{
                    'x':[self.train_data['x'][did] for did in train_cidxs[cid]], 'y':[self.train_data['y'][did] for did in train_cidxs[cid]]
                },
                # 'dvalid':{
                #     'x':[self.train_data['x'][did] for did in valid_cidxs[cid]], 'y':[self.train_data['y'][did] for did in valid_cidxs[cid]]
                # }
            }
        # with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
        #     ujson.dump(feddata, outf)
        with open(os.path.join(self.taskpath, 'data.pickle'), 'wb') as outf:
            pickle.dump(feddata, outf)
        return

    def IDXData_to_json(self, train_cidxs, valid_cidxs):
        if self.datasrc ==None:
            raise RuntimeError("Attr datasrc not Found. Please define it in __init__() before calling IndexData_to_json")
        feddata = {
            'store': 'IDX',
            'client_names': self.cnames,
            'dtest': [i for i in range(len(self.test_data))],
            'datasrc': self.datasrc
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain': train_cidxs[cid],
                'dvalid': valid_cidxs[cid]
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

class BasicTaskCalculator:

    _OPTIM = None

    def __init__(self, device):
        self.device = device
        self.lossfunc = None
        self.DataLoader = None

    def data_to_device(self, data):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_evaluation(self):
        raise NotImplementedError

    def get_data_loader(self, data, batch_size = 64):
        return NotImplementedError

    def test(self):
        raise NotImplementedError

    def get_optimizer(self, name="sgd", model=None, lr=0.1, weight_decay=0, momentum=0):
        # if self._OPTIM == None:
        #     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
        if name.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif name.lower() == 'adam':
            return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise RuntimeError("Invalid Optimizer.")

    @classmethod
    def setOP(cls, OP):
        cls._OPTIM = OP

class ClassifyCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(ClassifyCalculator, self).__init__(device)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def get_loss(self, model, data, device=None):
        tdata = self.data_to_device(data, device)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[1])
        return loss

    @torch.no_grad()
    def get_evaluation(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item()

    @torch.no_grad()
    def test(self, model, data, device=None):
        """Metric = Accuracy"""
        tdata = self.data_to_device(data, device)
        model = model.to(device)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[-1])
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item()

    def data_to_device(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, droplast=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast)

class BasicTaskReader:
    def __init__(self, taskpath=''):
        self.taskpath = taskpath

    def read_data(self):
        """
            Reading the spilted dataset from disk files and loading data into the class 'LocalDataset'.
            This algorithm should read three types of data from the processed task:
                train_sets = [client1_train_data, ...] where each item is an instance of 'LocalDataset'
                valid_sets = [client1_valid_data, ...] where each item is an instance of 'LocalDataset'
                test_set = test_dataset
            Return train_sets, valid_sets, test_set, client_names
        """
        pass

class XYTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(XYTaskReader, self).__init__(taskpath)

    def read_data(self, sample=False):
        if os.path.isfile(os.path.join(self.taskpath, 'data.json')):
            with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
                feddata = ujson.load(inf)
        elif os.path.isfile(os.path.join(self.taskpath, 'data.pickle')):
            with open(os.path.join(self.taskpath, 'data.pickle'), 'rb') as inf:
                feddata = pickle.load(inf)
        else:
            raise FileNotFoundError
                 
        test_data = XYDataset(feddata['dtest']['x'], feddata['dtest']['y'])
        valid_datas = XYDataset(feddata['dvalid']['x'], feddata['dvalid']['y'])
        if not sample:
            train_datas = [XYDataset(feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']) for name in feddata['client_names']]
        else:
            train_datas = [XYSampleDataset(feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']) for name in feddata['client_names']]
        # valid_datas = [XYDataset(feddata[name]['dvalid']['x'], feddata[name]['dvalid']['y']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

class IDXTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(IDXTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        DS = getattr(importlib.import_module(feddata['datasrc']['lib']), feddata['datasrc']['class_name'])
        arg_strings = '(' + ','.join(feddata['datasrc']['args'])
        train_args = arg_strings + ', train=True)'
        test_args = arg_strings + ', train=False)'
        DS.SET_DATA(eval(feddata['datasrc']['class_name'] + train_args))
        DS.SET_DATA(eval(feddata['datasrc']['class_name'] + test_args), key='TEST')
        test_data = IDXDataset(feddata['dtest'], key='TEST')
        train_datas = [IDXDataset(feddata[name]['dtrain']) for name in feddata['client_names']]
        valid_datas = [IDXDataset(feddata[name]['dvalid']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

class XYDataset(Dataset):
    def __init__(self, X=[], Y=[], totensor = True):
        """ Init Dataset with pairs of features and labels/annotations.
        XYDataset transforms data that is list\array into tensor.
        The data is already loaded into memory before passing into XYDataset.__init__()
        and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
        Args:
            X: a list of features
            Y: a list of labels with the same length of X
        """
        if not self._check_equal_length(X, Y):
            raise RuntimeError("Different length of Y with X.")
        if totensor:
            try:
                self.X = torch.tensor(X)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X = X
            self.Y = Y
        self.all_labels = list(set(self.tolist()[1]))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def tolist(self):
        if not isinstance(self.X, torch.Tensor):
            return self.X, self.Y
        return self.X.tolist(), self.Y.tolist()

    def _check_equal_length(self, X, Y):
        return len(X)==len(Y)

    def get_all_labels(self):
        return self.all_labels

class XYSampleDataset(XYDataset):
    def __init__(self, X=[], Y=[], totensor = True, percent=1.0, mode='exact', k=100):
        """ Init Dataset with pairs of features and labels/annotations.
        XYDataset transforms data that is list\array into tensor.
        The data is already loaded into memory before passing into XYDataset.__init__()
        and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
        Args:
            X: a list of features
            Y: a list of labels with the same length of X
        """
        super(XYSampleDataset, self).__init__(X, Y, totensor)
        self.mode = mode
        self.k = k
        num_samples, num_classes = len(X) ,len(self.all_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.cls_positive = collections.defaultdict(list)
        
        for i in range(num_samples):
            self.cls_positive[Y[i]].append(i)
        self.cls_negative = collections.defaultdict(list)
        for label in self.all_labels:
            for label_ in self.all_labels:
                if label == label_:
                    continue

                self.cls_negative[label].extend(self.cls_positive[label_])
        self.cls_positive = [np.asarray(self.cls_positive[self.idx_to_label[i]]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[self.idx_to_label[i]]) for i in range(num_classes)]
        
        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[self.idx_to_label[i]])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        img, target = self.X[index], self.Y[index]
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[self.label_to_idx[target.item()]], 1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)

        replace = True if self.k > len(self.cls_negative[self.label_to_idx[target.item()]]) else False
        try:
            neg_idx = np.random.choice(self.cls_negative[self.label_to_idx[target.item()]], self.k, replace=replace)
        except:
            return img, target
            
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        return img, target, index, sample_idx

    def tolist(self):
        if not isinstance(self.X, torch.Tensor):
            return self.X, self.Y
        return self.X.tolist(), self.Y.tolist()

    def _check_equal_length(self, X, Y):
        return len(X)==len(Y)

    def get_all_labels(self):
        return self.all_labels


class IDXDataset(Dataset):
    # The source dataset that can be indexed by IDXDataset
    _DATA = {'TRAIN': None,'TEST': None}

    def __init__(self, idxs, key='TRAIN'):
        """Init dataset with 'src_data' and a list of indexes that are used to position data in 'src_data'"""
        if not isinstance(idxs, list):
            raise RuntimeError("Invalid Indexes")
        self.idxs = idxs
        self.key = key

    @classmethod
    def SET_DATA(cls, dataset, key = 'TRAIN'):
        cls._DATA[key] = dataset

    @classmethod
    def ADD_KEY_TO_DATA(cls, key, value = None):
        if key==None:
            raise RuntimeError("Empty key when calling class algorithm IDXData.ADD_KEY_TO_DATA")
        cls._DATA[key]=value

    def __getitem__(self, item):
        idx = self.idxs[item]
        return self._DATA[self.key][idx]

class TupleDataset(Dataset):
    def __init__(self, X1=[], X2=[], Y=[], totensor=True):
        if totensor:
            try:
                self.X1 = torch.tensor(X1)
                self.X2 = torch.tensor(X2)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X1 = X1
            self.X2 = X2
            self.Y = Y

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.Y[item]

    def __len__(self):
        return len(self.Y)

    def tolist(self):
        if not isinstance(self.X1, torch.Tensor):
            return self.X1, self.X2, self.Y
        return self.X1.tolist(), self.X2.tolist(), self.Y.tolist()
