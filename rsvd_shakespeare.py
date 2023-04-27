import numpy as np

from sklearn.utils.extmath import randomized_svd


import rayleaf
from rayleaf.entities import Server, Client
from rayleaf.utils.logging_utils import log


def rsvd_shakespeare(
    output_dir: str,
    rank=-1,
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
    class CompClient(Client):
        def init(self):
            for k in self.model.state_dict():
                print(k, self.model.state_dict()[k].shape)


        def train(self, server_update):
            self.model_params = server_update
            grads = self.train_model(compute_grads=True)

            res = []
            for layer in grads.tensors:
                if rank > 0 and layer.shape == (1024, 256):
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
            log(f"Model layer shapes: {self.model_params.shapes}")


        def server_update(self):
            return self.model_params


        def update_model(self, client_updates):
            grads_decompressed = []

            for update in client_updates:
                grads_compressed = update["res"]
                decompressed = []
                for layer in grads_compressed:
                    if isinstance(layer, tuple):
                        U, S, Vt = layer
                        decompressed.append(np.dot(U * S, Vt))
                    else:
                        decompressed.append(layer)
                grads_decompressed.append(rayleaf.TensorArray(decompressed))

            average_grads = 0
            total = 0
            
            for i, update in enumerate(client_updates):
                average_grads += grads_decompressed[i] * update["n"]
                total += update["n"]

            average_grads /= total
            return self.model_params + average_grads


    rayleaf.run_experiment(
        dataset = "shakespeare",
        dataset_dir = "data/shakespeare/",
        output_dir= output_dir,
        model = "stacked_lstm",
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
