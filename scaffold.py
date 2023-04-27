from datetime import datetime


import rayleaf
from rayleaf.entities import Server, Client


def scaffold(
    dataset: str,
    global_lr: float,
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
    use_grads = False,
    notes: str = None
):
    curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    class ScaffoldClient(Client):
        def init(self):
            self.c_i_plus = 0


        def train(self, server_update):
            self.model_params = server_update["x"]

            self.c = server_update["c"]
            self.c_i = self.c_i_plus

            grads = self.train_model(compute_grads=True)

            self.c_i_plus = self.c_i - self.c + (1 / (self.num_epochs * client_lr)) \
                * (server_update["x"] - self.model_params)

            return {
                "dy_i": grads,
                "dc_i": self.c_i_plus - self.c_i
            }

        
        def run_minibatch(self, X, y):
            Client.run_minibatch(self, X, y)
            self.model_params = self.model_params + client_lr * self.c_i - client_lr * self.c


    class ScaffoldServer(Server):
        def init(self):
            self.c = 0


        def server_update(self):
            return {
                "x": self.model_params,
                "c": self.c
            }

        
        def update_model(self, client_updates):
            S = len(client_updates)
            dx, dc = 0, 0

            for update in client_updates:
                dx += update["dy_i"]
                dc += update["dc_i"]
            
            dx /= S
            dc /= S

            self.c += (S / num_clients) * dc

            return self.model_params + global_lr * dx


    if dataset == "femnist":
        model = "cnn"
    elif dataset == "speech_commands":
        model = "m5"

    rayleaf.run_experiment(
        dataset = dataset,
        dataset_dir = f"data/{dataset}/",
        output_dir= f"output/scaffold-{dataset}-{curr_time}/",
        model = model,
        num_rounds = num_rounds,
        eval_every = eval_every,
        ServerType=ScaffoldServer,
        client_types=[(ScaffoldClient, num_clients)],
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
