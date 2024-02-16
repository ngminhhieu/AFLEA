import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing

class MyLogger(flw.Logger):
    def log(self, server=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "mean_curve":[],
                "var_curve":[],
                "train_losses":[],
                "test_accs":[],
                "test_losses":[],
                "valid_accs":[],
                # "client_accs":{},
                # "mean_valid_accs":[],
            }
        if "mp_" in server.name:
            valid_metrics, valid_losses = server.test(split='val', device=torch.device('cuda:0'))
        else:
            valid_metrics, valid_losses = server.test(split='val')

        # train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        # self.output['train_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)])/server.data_vol)
        # self.output['mean_valid_accs'].append(1.0*sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)])/server.data_vol)
        self.output['valid_accs'].append(valid_metrics)
        # self.output['test_losses'].append(test_loss)
        # self.output['mean_curve'].append(np.mean(valid_metrics,0).tolist()) 
        # self.output['var_curve'].append(np.std(valid_metrics,0).tolist()) 
        
        # for cid in range(server.num_clients):
        #     self.output['client_accs'][server.clients[cid].name]=[self.output['valid_accs'][i][cid] for i in range(len(self.output['valid_accs']))]
    
        # print("Training Loss:", self.output['train_losses'][-1])    
        # print("Testing Loss:", self.output['test_losses'][-1])
        # print("Testing Accuracy:", self.output['test_accs'][-1]) 
        print("Validating Accuracy:", self.output['valid_accs'][-1])
        # print("Mean of Client Accuracy:", self.output['mean_curve'][-1])
        # print("Std of Client Accuracy:", self.output['var_curve'][-1])
        #if self.output['valid_accs'][-1] >= self.best_val_acc:
        #    self.best_val_acc = self.output['valid_accs'][-1] 
        if "mp_" in server.name:
            self.best_test_acc, self.best_test_loss = server.test(split='test', device=torch.device('cuda:0'))
        else:
            self.best_test_acc, self.best_test_loss = server.test(split='test')

        print("Overall Testing Accuracy:", self.best_test_acc)
        print("Overall Testing Loss:", self.best_test_loss)
        self.output['test_accs'].append(self.best_test_acc)
        self.output['train_losses'].append(self.best_test_loss)
        
logger = MyLogger()

def main():
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




