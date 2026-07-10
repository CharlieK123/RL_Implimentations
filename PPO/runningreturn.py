import torch


class RunningNorm:
    def __init__(self):
        self.running_mean = torch.tensor(0.0)
        self.running_var = torch.tensor(1.0)
        self.count = 1e-4


    def update(self, x):
        x = x.detach().float()

        # update mean
        sample_mean = torch.mean(x)
        batch_count = x.numel()

        delta_mean = sample_mean - self.running_mean
        total_count = batch_count + self.count
        self.running_mean += delta_mean * (batch_count / total_count)

        # update var
        sample_var = torch.var(x, unbiased=False)
        old_m2 = self.running_var * self.count
        batch_m2 = sample_var * batch_count

        correction = torch.square(delta_mean) * self.count * batch_count / total_count
        new_m2 = old_m2 + batch_m2 + correction

        self.running_var = new_m2 / total_count
        self.count = total_count

    @property
    def std(self):
        return torch.sqrt(self.running_var + 1e-8)

    def normalize(self, x):
        return (x - self.running_mean) / self.std