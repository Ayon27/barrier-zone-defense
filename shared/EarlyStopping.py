import torch
import numpy as np


class EarlyStopping:
    def __init__(
        self,
        path,
        patience,
        delta=0,
        trace_func=print,
    ):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.is_stopping_early = False
        self.loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, loss, model):
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"   [EarlyStopping] Counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.is_stopping_early = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        self.trace_func(
            f"   [EarlyStopping] Loss decreased ({self.loss_min:.6f} --> {loss:.6f}). Saving model..."
        )
        torch.save(model.state_dict(), self.path)
        self.loss_min = loss
