from collections import OrderedDict
from datetime import datetime


import torch


import rayleaf
from rayleaf.entities import Server, Client


curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def make_sign_server_client(server_lr: float, momentum: float = 0):
    class SignClient(Client):
        def train(self):
            self.train_model(compute_grads=True)

            updates = OrderedDict()

            for param_tensor, grad in self.grads.items():
                if "running_mean" not in param_tensor and "running_var" not in param_tensor:
                    updates[param_tensor] = torch.sign(grad)
                else:
                    updates[param_tensor] = torch.zeros(grad.shape)

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
                self.model_params[param_tensor] = (self.model_params[param_tensor] + update).to(self.model_params[param_tensor].dtype)
    

    return SignServer, SignClient


SignServer, SignClient = make_sign_server_client(server_lr=0.005, momentum=0.9)

rayleaf.run_experiment(
    dataset = "speech_commands",
    dataset_dir = "datasets/speech_commands/",
    output_dir=f"output/debug/signavg-speech_commands-{curr_time}/",
    model = "m5",
    num_rounds = 100,
    eval_every = 10,
    ServerType=Server,
    client_types=[(Client, 10)],
    clients_per_round = 5,
    client_lr = 0.06,
    batch_size = 64,
    seed = 0,
    use_val_set = False,
    num_epochs = 5,
    gpus_per_client_cluster=0.1,
    num_client_clusters=2,
    save_model=False,
    notes = f"server_lr = 0.005, momentum = 0.9"
)
