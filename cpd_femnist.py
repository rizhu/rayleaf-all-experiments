from datetime import datetime
from pathlib import Path


import numpy as np
import tensorly as tl

from sklearn.utils.extmath import randomized_svd
from tensorly.decomposition import parafac


import rayleaf
from rayleaf.entities import Server, Client
from rayleaf.utils.logging_utils import log


def cpd_femnist(
    output_dir: str,
    conv1_rank=-1,
    conv2_rank=-1,
    fc1_rank=-1,
    fc2_rank=-1,
    fedprox=False,
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


    class CompFedAvgClient(Client):
        def init(self):
            self.delete_model_on_completion = True

        
        def train(self, server_update):
            self.model_params = server_update
            self.train_model(compute_grads=False)

            res = []
            for layer in self.model_params.tensors:
                if conv1_rank > 0 and layer.shape == (32, 1, 5, 5):
                    factors = parafac(tl.tensor(layer), rank=conv1_rank)
                    res.append(factors)
                elif conv2_rank > 0 and layer.shape == (64, 32, 5, 5):
                    factors = parafac(tl.tensor(layer), rank=conv2_rank)
                    res.append(factors)
                elif fc1_rank > 0 and layer.shape == (2048, 3136):
                    layer = layer.detach().numpy()
                    U, S, Vt = randomized_svd(layer, n_components=fc1_rank, n_iter="auto", random_state=None)

                    res.append((U, S, Vt))
                elif fc2_rank > 0 and layer.shape == (62, 2048):
                    layer = layer.detach().numpy()
                    U, S, Vt = randomized_svd(layer, n_components=fc2_rank, n_iter="auto", random_state=None)

                    res.append((U, S, Vt))
                else:
                    res.append(layer)

            return {
                "res": res,
                "n": self.num_train_samples
            }


    class CompFedProxClient(Client):
        def init(self):
            self.delete_model_on_completion = True

        
        def train(self, server_update):
            self.server_weights = server_update
            self.model_params = server_update
            self.train_model(compute_grads=False)
            del self.server_weights

            res = []
            for layer in self.model_params.tensors:
                if conv1_rank > 0 and layer.shape == (32, 1, 5, 5):
                    factors = parafac(tl.tensor(layer), rank=conv1_rank)
                    res.append(factors)
                elif conv2_rank > 0 and layer.shape == (64, 32, 5, 5):
                    factors = parafac(tl.tensor(layer), rank=conv2_rank)
                    res.append(factors)
                elif fc1_rank > 0 and layer.shape == (2048, 3136):
                    layer = layer.detach().numpy()
                    U, S, Vt = randomized_svd(layer, n_components=fc1_rank, n_iter="auto", random_state=None)

                    res.append((U, S, Vt))
                elif fc2_rank > 0 and layer.shape == (62, 2048):
                    layer = layer.detach().numpy()
                    U, S, Vt = randomized_svd(layer, n_components=fc2_rank, n_iter="auto", random_state=None)

                    res.append((U, S, Vt))
                else:
                    res.append(layer)

            return {
                "res": res,
                "n": self.num_train_samples
            }
        

        def compute_loss(self, probs, targets):
            return self.model.loss_fn(probs, targets) + (1 / 2) * np.linalg.norm(self.server_weights - self.model_params)



    class CompServer(Server):
        def init(self):
            log(f"Model layer shapes: {self.model_params.shapes}")


        def server_update(self):
            return self.model_params


        def update_model(self, client_updates):
            models_decompressed = []

            for update in client_updates:
                grads_compressed = update["res"]
                decompressed = []
                for layer in grads_compressed:
                    if isinstance(layer, tuple):
                        U, S, Vt = layer
                        decompressed.append(np.dot(U * S, Vt))
                    elif isinstance(layer, tl.cp_tensor.CPTensor):
                        decompressed.append(tl.cp_to_tensor(layer))
                    else:
                        decompressed.append(layer)
                models_decompressed.append(rayleaf.TensorArray(decompressed))

            average_params = 0
            total = 0
            
            for i, update in enumerate(client_updates):
                average_params += models_decompressed[i] * update["n"]
                total += update["n"]

            average_params /= total
            return average_params


    client_type = CompFedProxClient if fedprox else CompFedAvgClient


    rayleaf.run_experiment(
        dataset = "femnist",
        dataset_dir = "data/femnist/",
        output_dir= Path(output_dir, "femnist", "cpd", f"{fedprox}-{num_clients}clients-{clients_per_round}cpr-{client_lr}lr-{num_epochs}epochs-{num_rounds}rounds"),
        model = "cnn",
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=CompServer,
        client_types=[(client_type, num_clients)],
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
