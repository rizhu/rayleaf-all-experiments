from datetime import datetime
from pathlib import Path


from numpy.linalg import norm


import rayleaf
from rayleaf.entities import Server, Client


def fedprox(
    dataset: str,
    output_dir: str,
    mu: float,
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

    class FedProxClient(Client):
        def init(self):
            self.delete_model_on_completion = False


        def train(self, server_update):
            self.server_weights = server_update
            res = Client.train(self, server_update)
            del self.server_weights

            return res


        def compute_loss(self, probs, targets):
            return self.model.loss_fn(probs, targets) + (mu / 2) * norm(self.server_weights - self.model_params)
        

    if dataset == "femnist":
        model = "cnn"
    elif dataset == "speech_commands":
        model = "m5"
    elif dataset == "shakespeare":
        model = "stacked_lstm"

    rayleaf.run_experiment(
        dataset = dataset,
        dataset_dir = f"data/{dataset}/",
        output_dir=Path(output_dir, dataset, "fedprox", f"{num_clients}clients-{clients_per_round}cpr-{client_lr}lr-{num_epochs}epochs-{num_rounds}rounds-{mu}mu"),
        model = model,
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=Server,
        client_types=[(FedProxClient, num_clients)],
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
