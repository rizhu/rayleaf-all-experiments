from datetime import datetime

import numpy as np
import torch


import rayleaf
from rayleaf.entities import Server, Client


def dpsgd_cnn(
    stdev: float,
    C: float,
    num_rounds: int = 100,
    eval_every: int = 10,
    num_clients: int = 200,
    clients_per_round: int = 40,
    client_lr: float = 0.05,
    batch_size: int = 64,
    seed: int = 0,
    num_epochs: int = 10,
    gpus_per_client_cluster: float = 1,
    num_client_clusters: int = 8,
    save_model: bool = False,
    notes: str = None
):
    curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    class DPSGDClient(Client):
        def train(self):
        # def train(self, server_update):
            self.train_model(compute_grads=True)

            self.collect_metric([torch.linalg.norm(g).item() for g in self.grads], "norms")

            for i, layer in enumerate(self.grads):
                self.grads[i] /= max(1, torch.linalg.norm(layer) / C)
                self.grads[i] += stdev * C * torch.randn(layer.shape)

            return self.grads
            # self.model_params = server_update
            # grads = self.train_model(compute_grads=True)

            # grads /= max(1, np.linalg.norm(grads) / C)

            
    

    class DPSGDServer(Server):
        def update_layer(self, current_params, updates: list, client_num_samples: list, num_clients: int):
            average_grads = 0
            for i in range(num_clients):
                average_grads += updates[i] * client_num_samples[i]
            
            average_grads /= self.num_train_samples

            return current_params + average_grads


    rayleaf.run_experiment(
        dataset = "femnist",
        dataset_dir = "data/femnist/",
        output_dir= f"output/dpsgd_cnn-{curr_time}/",
        model = "cnn",
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=DPSGDServer,
        client_types=[(DPSGDClient, num_clients)],
        clients_per_round = clients_per_round,
        client_lr = client_lr,
        batch_size = batch_size,
        seed = seed,
        use_val_set = False,
        num_epochs = num_epochs,
        gpus_per_client_cluster = gpus_per_client_cluster,
        num_client_clusters = num_client_clusters,
        save_model = save_model,
        notes = notes
    )
