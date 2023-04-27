from collections import OrderedDict
from datetime import datetime

import torch


import rayleaf
from rayleaf.entities import Server, Client


def sign_femnist(
    server_lr=0.005,
    momentum=0.9,
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

    def make_signavg_server_client():
        class SignClient(Client):
            def train(self):
                self.train_model(compute_grads=True)

                updates = OrderedDict()

                for param_tensor, grad in self.grads.items():
                    updates[param_tensor] = torch.sign(grad)

                return self.num_train_samples, updates
        

        class SignServer(Server):
            def init(self):
                self.mo = OrderedDict()
                for param_tensor, layer in self.model_params.items():
                    self.mo[param_tensor] = torch.zeros(layer.shape)

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
        

        return SignServer, SignClient


    SignServer, SignClient = make_signavg_server_client()

    rayleaf.run_experiment(
        dataset = "femnist",
        dataset_dir = "data/femnist/",
        output_dir= f"output/sign_femnist-{curr_time}/",
        model = "cnn",
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=SignServer,
        client_types=[(SignClient, num_clients)],
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
