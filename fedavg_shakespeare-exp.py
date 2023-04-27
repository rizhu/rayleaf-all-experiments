from collections import (
    OrderedDict
)

import torch


import rayleaf
from rayleaf.entities import Server, Client


NUM_ROUNDS = 100
EVAL_EVERY = 10
CLIENTS_PER_ROUND = 20
CLIENT_LR = 0.05
BATCH_SIZE = 64
SEED = 0
NUM_EPOCHS = 10


def make_signavg_server_client(server_lr: float, momentum: float = 0):
    class SignavgClient(Client):
        def train(self):
            self.train_model(compute_grads=True)

            updates = OrderedDict()

            for param_tensor, grad in self.grads.items():
                updates[param_tensor] = torch.sign(grad).cuda()

            return self.num_train_samples, updates
    

    class SignavgServer(Server):
        def init(self):
            self.mo = OrderedDict()
            for param_tensor, layer in self.model_params.items():
                self.mo[param_tensor] = torch.zeros(layer.shape).cuda()

        def update_model(self):
            self.reset_grads()

            total_weight = 0
            for (client_samples, client_grad) in self.updates:
                total_weight += client_samples

                for param_tensor, sign_grad in client_grad.items():
                    self.grads[param_tensor] += sign_grad * client_samples

            for param_tensor in self.grads.keys():
                self.grads[param_tensor] /= total_weight
            
            for param_tensor, layer in self.grads.items():
                self.mo[param_tensor] = momentum * self.mo[param_tensor] + server_lr * layer
            
            for param_tensor, update in self.mo.items():
                self.model_params[param_tensor] += update
    

    return SignavgServer, SignavgClient

SignavgServer, SignavgClient = make_signavg_server_client(server_lr=0.005, momentum=0.9)
rayleaf.run_experiment(
    dataset = "shakespeare",
    dataset_dir = "data/shakespeare/",
    output_dir="output/fedavg_shakespeare_100rds/",
    model = "stacked_lstm",
    num_rounds = NUM_ROUNDS,
    eval_every = EVAL_EVERY,
    ServerType=Server,
    client_types=[(Client, -1)],
    clients_per_round = CLIENTS_PER_ROUND,
    client_lr = CLIENT_LR,
    batch_size = BATCH_SIZE,
    seed = SEED,
    use_val_set = False,
    num_epochs = NUM_EPOCHS,
    gpus_per_client_cluster=0.6,
    num_client_clusters=7,
    save_model=False
)
