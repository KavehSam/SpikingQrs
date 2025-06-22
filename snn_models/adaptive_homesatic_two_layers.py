import snntorch as snn
import torch
import torch.nn as nn



class AdaptiveHomeostasisTwoLayers(nn.Module):
    def __init__(
        self,
        config=None,
        input_size: int = 10,
        hidden_size: int = 10,
        output_size: int = 1,
        num_time_steps: int = 100,
        target_rate: float = 0.02,
        seed: int = 42,
    ):
        super().__init__()

        # Load configuration if provided
        if config is not None:
            network_config = config.get_network_config()
            learning_config = config.get_learning_config()
            stdp_config = config.get_stdp_config()
            homeostasis_config = config.get_homeostasis_config()
            
            input_size = network_config.get("input_size", input_size)
            hidden_size = network_config.get("hidden_size", hidden_size)
            output_size = network_config.get("output_size", output_size)
            num_time_steps = network_config.get("num_time_steps", num_time_steps)
            target_rate = learning_config.get("target_rate", target_rate)
            seed = network_config.get("seed", seed)
        
        torch.manual_seed(seed)

        # Network Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_time_steps = num_time_steps
        self.batch_size = 1
        self.target_rate = target_rate
        self.homeo_scale = 0.02
        self.learning_rate = learning_config.get("learning_rate", 0.001) if config else 0.001
        self.base_threshold = 1.0

        # STDP Parameters
        self.A_plus = torch.tensor(stdp_config.get("A_plus", 0.015), dtype=torch.float32) if config else torch.tensor(0.015, dtype=torch.float32)
        self.A_minus = torch.tensor(stdp_config.get("A_minus", 0.012), dtype=torch.float32) if config else torch.tensor(0.012, dtype=torch.float32)
        self.tau_plus = torch.tensor(stdp_config.get("tau_plus", 20.0), dtype=torch.float32) if config else torch.tensor(20.0, dtype=torch.float32)
        self.tau_minus = torch.tensor(stdp_config.get("tau_minus", 20.0), dtype=torch.float32) if config else torch.tensor(20.0, dtype=torch.float32)

        # Homeostasis Parameters
        self.homeostasis_params = {
            "hidden": {"scale": homeostasis_config.get("alpha", 0.015), "range": (0.8, 4.0), "decay": 0.99},
            "output": {"scale": homeostasis_config.get("beta", 0.02), "range": (0.5, 3.0), "decay": 0.995},
        }
        # Target Firing Rates for Homeostasis
        self.target_rate_min = learning_config.get("target_rate_min", 0.01) if config else 0.01  # 1% minimum
        self.target_rate_max = learning_config.get("target_rate_max", 0.05) if config else 0.05  # 5% maximum

        # Construct Network
        self.construct_network()
        self.reset_network()

    def set_num_time_steps(self, num_time_steps):
        self.num_time_steps = num_time_steps

    def construct_network(self):
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.lif1 = snn.Leaky(
            beta=0.9, threshold=self.base_threshold, learn_threshold=True
        )
        self.lif2 = snn.Leaky(
            beta=0.9, threshold=self.base_threshold, learn_threshold=True
        )

    def forward(self, x):
        self.inpu_spikes = x
        for step in range(self.num_time_steps):
            cur1 = self.fc1(self.inpu_spikes[step])
            spk1, self.mem1 = self.lif1(cur1, self.mem1)
            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)

            self.spk1_rec.append(spk1)
            self.spk2_rec.append(spk2)

        return torch.stack(self.spk1_rec), torch.stack(self.spk2_rec)

    def init_traces(self, pre_size, post_size):
        return {
            "pre": torch.zeros(pre_size, dtype=torch.float32),
            "post": torch.zeros(post_size, dtype=torch.float32),
        }

    def reset_network(self):
        # Initialize STDP traces
        self.trace1 = self.init_traces(self.input_size, self.hidden_size)
        self.trace2 = self.init_traces(self.hidden_size, self.output_size)

        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")

        # Weight normalization
        with torch.no_grad():
            self.fc1.weight.data = torch.renorm(
                self.fc1.weight, p=2, dim=0, maxnorm=1.5
            )
            self.fc2.weight.data = torch.renorm(
                self.fc2.weight, p=2, dim=0, maxnorm=1.0
            )

        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

        self.spk1_rec = []
        self.spk2_rec = []

    def update_weights(self, spk1_rec: torch.Tensor, spk2_rec: torch.Tensor) -> None:
        """Update weights of the network
        Using STDP learning rule with adaptive homeostasis.

        Args:
            spk1_rec (torch.Tensor): Spike train of the first layer.
            spk2_rec (torch.Tensor): Spike train of the second layer.
        """
        # STDP Learning for input-hidden layer
        for t in range(self.num_time_steps):
            ### STDP Learning
            # Pre and post decay
            decay_pre = torch.exp(
                torch.tensor(-1.0, dtype=torch.float32) / self.tau_plus
            )
            decay_post = torch.exp(
                torch.tensor(-1.0, dtype=torch.float32) / self.tau_minus
            )

            ## input-hidden layer
            # Update traces
            self.trace1["pre"] = (
                self.trace1["pre"] * decay_pre
                + self.inpu_spikes[t].squeeze() * self.A_plus
            )
            self.trace1["post"] = (
                self.trace1["post"] * decay_post + spk1_rec[t].squeeze() * self.A_minus
            )

            # Calculate weight updates
            delta_w1 = torch.outer(self.trace1["post"], self.trace1["pre"])
            self.fc1.weight.data += self.learning_rate * delta_w1

            ## hidden-output layer
            # Update traces
            self.trace2["pre"] = (
                self.trace2["pre"] * decay_pre + spk1_rec[t].squeeze() * self.A_plus
            )
            self.trace2["post"] = (
                self.trace2["post"] * decay_post + spk2_rec[t].squeeze() * self.A_minus
            )

            # Calculate weight updates
            delta_w2 = torch.outer(self.trace2["post"], self.trace2["pre"])
            self.fc2.weight.data += self.learning_rate * delta_w2

        # Adaptive homeostasis for output layer
        with torch.no_grad():
            self.firing_rates1 = torch.mean(spk1_rec)
            self.firing_rates2 = torch.mean(spk2_rec)
            # Output layer homeostasis regulation
            if self.firing_rates2 < self.target_rate_min:
                adj = self.homeostasis_params["output"]["scale"] * (
                    self.target_rate_min - self.firing_rates2
                )
                self.lif2.threshold -= adj
            elif self.firing_rates2 > self.target_rate_max:
                adj = (
                    self.homeostasis_params["output"]["scale"]
                    * 2
                    * (self.firing_rates2 - self.target_rate_max)
                )
                self.lif2.threshold += adj

            self.lif2.threshold.data = (
                self.homeostasis_params["output"]["decay"] * self.lif2.threshold
                + (1 - self.homeostasis_params["output"]["decay"]) * self.base_threshold
            )
            self.lif2.threshold.clamp_(*self.homeostasis_params["output"]["range"])

            # Hidden layer homeostasis regulation
            if self.firing_rates1 < self.target_rate_min:
                adj = self.homeostasis_params["output"]["scale"] * (
                    self.target_rate_min - self.firing_rates1
                )
                self.lif1.threshold -= adj
            elif self.firing_rates1 > self.target_rate_max:
                adj = (
                    self.homeostasis_params["output"]["scale"]
                    * 2
                    * (self.firing_rates1 - self.target_rate_max)
                )
                self.lif1.threshold += adj

            self.lif1.threshold.data = (
                self.homeostasis_params["output"]["decay"] * self.lif1.threshold
                + (1 - self.homeostasis_params["output"]["decay"]) * self.base_threshold
            )
            self.lif1.threshold.clamp_(*self.homeostasis_params["output"]["range"])

        # Activity-dependent STDP scaling of weights
        stdp_scale = torch.sigmoid(10 * (self.target_rate_max - self.firing_rates2))
        self.fc2.weight.data *= stdp_scale
