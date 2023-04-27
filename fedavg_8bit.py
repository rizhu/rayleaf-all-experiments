import numpy as np
import torch

from datetime import datetime
from pathlib import Path


import rayleaf
from rayleaf.entities import Server, Client
from rayleaf.entities.constants import MODEL_PARAMS_KEY, NUM_SAMPLES_KEY


def fedavg_8bit(
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

    class SmallClient(Client):
        def init(self):
            self.delete_model_on_completion = True

        
        def train(self, server_update):
            self.model_params = server_update["params"]
            self.train_model()

            truncated_params = self.model_params

            if server_update["round"] > 50:
                # Get the exponents E = floor( log_2 |X| ) + 127
                exponents = np.floor(np.log2(np.abs(self.model_params))) + 127
                # Get the mantissas M = X / 2^{E - 127}
                mantissa = np.abs(self.model_params) / np.power(2, exponents - 127)
                # Truncate mantissas to just top 3 bits: M = int[ (M - 1) * 2^3 ] / 2^3 + 1
                mantissa = ((mantissa - 1) * 8).to(int) / 8 + 1
                # Truncate exponents to top 4 bits: E = E // 16 * 16
                exponents = exponents // 16 * 16
                # Build truncated X: X_trunc = sign(X) * 2^{E - 127} * M
                truncated_params = np.sign(self.model_params) * np.power(2, exponents - 127) * mantissa

            return {
                MODEL_PARAMS_KEY: truncated_params,
                NUM_SAMPLES_KEY: self.num_train_samples
            }


    class SwitchServer(Server):
        def server_update(self):
            return {
                "params": self.model_params,
                "round": self.curr_round
            }


    if dataset == "femnist":
        model = "cnn"
    elif dataset == "speech_commands":
        model = "m5"
    elif dataset == "shakespeare":
        model = "stacked_lstm"

    rayleaf.run_experiment(
        dataset = dataset,
        dataset_dir = f"data/{dataset}/",
        output_dir= Path(output_dir, dataset, "fedavg", f"{num_clients}clients-{clients_per_round}cpr-{client_lr}lr-{num_epochs}epochs-{num_rounds}rounds-8bit"),
        model = model,
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=SwitchServer,
        client_types=[(SmallClient, num_clients)],
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
