from datetime import datetime
from pathlib import Path


import numpy as np

from sklearn.utils.extmath import randomized_svd


import rayleaf
from rayleaf.entities import Server, Client


COMP_SHAPES = {
    "femnist": (2048, 3136),
    "shakespeare": (1024, 256)
}


def fedprox_rsvd(
    dataset: str,
    output_dir: str,
    mu: int,
    rank: int = -1,
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

    class CompClient(Client):
        def init(self):
            self.delete_model_on_completion = True


        def train(self, server_update):
            self.model_params = server_update
            self.server_weights = server_update
            self.train_model(compute_grads=True)
            del self.server_weights

            res = []
            for layer in self.model_params.tensors:
                if rank > 0 and layer.shape == COMP_SHAPES.get(dataset, None):
                    try:
                        layer = layer.detach().numpy()
                        U, S, Vt = randomized_svd(layer, n_components=rank, n_iter="auto", random_state=None)

                        res.append((U, S, Vt))
                    except:
                        res.append(layer)
                else:
                    res.append(layer)

            return {
                "res": res,
                "n": self.num_train_samples
            }


        def compute_loss(self, probs, targets):
            return self.model.loss_fn(probs, targets) + (mu / 2) * np.linalg.norm(self.server_weights - self.model_params)


    class CompServer(Server):
        def init(self):
            print("Model Architecture:", self.model_params.shapes)


        def server_update(self):
            return self.model_params


        def update_model(self, client_updates):
            models_decompressed = []

            for update in client_updates:
                model_compressed = update["res"]
                decompressed = []
                for layer in model_compressed:
                    if isinstance(layer, tuple):
                        U, S, Vt = layer
                        decompressed.append(np.dot(U * S, Vt))
                    else:
                        decompressed.append(layer)
                models_decompressed.append(rayleaf.TensorArray(decompressed))

            average_model = 0
            total = 0
            
            for i, update in enumerate(client_updates):
                average_model += models_decompressed[i] * update["n"]
                total += update["n"]

            average_model /= total
            return average_model

    if dataset == "femnist":
        model = "cnn"
    elif dataset == "speech_commands":
        model = "m5"
    elif dataset == "shakespeare":
        model = "stacked_lstm"


    rayleaf.run_experiment(
        dataset = dataset,
        dataset_dir = f"data/{dataset}/",
        output_dir= Path(output_dir, dataset, "fedprox", f"r{str(rank).zfill(4)}-{num_clients}clients-{clients_per_round}cpr-{client_lr}lr-{num_epochs}epochs-{num_rounds}rounds-{mu}mu"),
        model = model,
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
