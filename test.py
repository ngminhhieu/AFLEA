import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing
import torch.nn.functional as F 

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
    model = server.get_model()
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    norm1, norm2, diff, sim = [], [], [], []
    with torch.no_grad():
        for client in server.clients:
            data_loader = client.calculator.get_data_loader(client.train_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                emb1, emb2 = model.get_intermediate(batch_data[0].to(device))
                emb1 = emb1.reshape((emb1.shape[0], -1))
                emb2 = emb2.reshape((emb2.shape[0], -1))

                norm1 += torch.sqrt((emb1**2).sum(-1)).cpu().numpy().tolist()
                norm2 += torch.sqrt((emb2**2).sum(-1)).cpu().numpy().tolist()
                diff += torch.sqrt(((emb1 - emb2)**2).sum(-1)).cpu().numpy().tolist()
                sim += F.cosine_similarity(emb1, emb2).cpu().numpy().tolist()

    print(np.mean(norm1), np.mean(norm2), np.mean(diff), np.mean(sim))

if __name__ == '__main__':
    main()




