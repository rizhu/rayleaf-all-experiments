from datetime import datetime


import numpy as np
import torch

from numpy.linalg import svd
from scipy.linalg import qr
from sklearn.utils.extmath import randomized_svd


import rayleaf
from rayleaf.entities import Server, Client


def nla_cnn(
    output_dir: str,
    rank: int,
    arch_aware: bool = True,
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

    def make_arch_aware():
        class CompClient(Client):
            def train(self, server_update):
                self.model_params = server_update
                grads = self.train_model(compute_grads=True)

                res = []
                for layer in grads.tensors:
                    if rank > 0 and layer.shape == (2048, 3136):
                        layer = layer.detach().numpy()
                        U, S, Vt = randomized_svd(layer, n_components=rank, n_iter="auto", random_state=None)

                        res.append((U, S, Vt))
                    else:
                        res.append(layer)

                return {
                    "res": res,
                    "n": self.num_train_samples
                }
                    
        

        class CompServer(Server):
            def init(self):
                print("Number of parameters:", self.model_params.size)


            def server_update(self):
                return self.model_params


            def update_model(self, client_updates):
                grads_decompressed = []
                comp_layer = -1
                for update in client_updates:
                    grads_compressed = update["res"]
                    if comp_layer == -1:
                        for i, layer in enumerate(grads_compressed):
                            if type(layer) == tuple:
                                comp_layer = i
                    U, S, Vt = grads_compressed[comp_layer]
                    grads_compressed[comp_layer] = torch.Tensor(np.dot(U * S, Vt))
                    grads_decompressed.append(rayleaf.TensorArray(grads_compressed))

                average_grads = 0
                total = 0
                
                for i, update in enumerate(client_updates):
                    average_grads += grads_decompressed[i] * update["n"]
                    total += update["n"]

                average_grads /= total
                return self.model_params + average_grads


        return CompServer, CompClient
    
    def make_arch_unaware():
        class CompClient(Client):
            def init(self):
                self.dim0 = 1310


            def train(self, server_update):
                self.model_params = server_update
                grads = self.train_model(compute_grads=True)

                squarey = grads.flat().reshape(self.dim0, -1).numpy()
                U, S, Vt = randomized_svd(squarey, n_components=rank, n_iter="auto", random_state=None)

                return {
                    "res": (U, S, Vt),
                    "n": self.num_train_samples
                }
        

        class CompServer(Server):
            def server_update(self):
                return self.model_params


            def update_model(self, client_updates):
                average_grads = 0
                total = 0
                
                for update in client_updates:
                    U, S, Vt = update["res"]
                    average_grads += rayleaf.TensorArray.unflatten(np.dot(U * S, Vt).flatten(), self.model_params.shapes) * update["n"]
                    total += update["n"]

                average_grads /= total
                return self.model_params + average_grads


        return CompServer, CompClient

    if arch_aware:    
        CompServer, CompClient = make_arch_aware()
    else:
        CompServer, CompClient = make_arch_unaware()


    rayleaf.run_experiment(
        dataset = "femnist",
        dataset_dir = "data/femnist/",
        output_dir= output_dir,
        model = "cnn",
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=CompServer,
        client_types=[(CompClient, num_clients)],
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
