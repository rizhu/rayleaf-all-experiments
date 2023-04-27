import bz2
import pickle
import sys

from datetime import datetime
from pathlib import Path


import rayleaf
from rayleaf.entities import Server, Client
from rayleaf.utils.logging_utils import log


def comp_bz2(
    dataset: str,
    output_dir: str,
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

    class Bz2Client(Client):
        def train(self, server_update):
            Client.train(self, server_update)
            compressed = bz2.compress(pickle.dumps(self.model_params))
            self.collect_metric(sys.getsizeof(compressed), "compressed model")
            return {
                "compressed": compressed,
                "n": self.num_train_samples
            }


    class Bz2Server(Server):
        def init(self):
            log(f"Model layer shapes: {self.model_params.shapes}")


        def update_model(self, client_updates):
            num_samples = 0
            average_params = 0

            for update in client_updates:
                model = pickle.loads(bz2.decompress(update["compressed"]))
                average_params += model * update["n"]
                num_samples += update["n"]
            
            average_params /= num_samples
            return average_params


    if dataset == "femnist":
        model = "cnn"
    elif dataset == "speech_commands":
        model = "m5"
    elif dataset == "shakespeare":
        model = "stacked_lstm"

    rayleaf.run_experiment(
        dataset = dataset,
        dataset_dir = f"data/{dataset}/",
        output_dir= Path(output_dir, dataset, "bz2", f"{num_clients}clients-{clients_per_round}cpr-{client_lr}lr-{num_epochs}epochs-{num_rounds}rounds"),
        model = model,
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=Bz2Server,
        client_types=[(Bz2Client, num_clients)],
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
