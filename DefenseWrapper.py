import torch
import torch.nn as nn


class DefenseWrapper(nn.Module):
    """Adapter to wrap BARZ to behave like a standard torch model for attacks."""

    def __init__(self, defense, device):
        super().__init__()
        self.defense = defense
        self.device = device

    def forward(self, x):
        sampleSize = x.shape[0]
        modelVotes = torch.zeros(
            (self.defense.ModelNum, sampleSize, self.defense.ClassNum)
        ).to(self.device)
        for i in range(self.defense.ModelNum):
            pred = self.defense.ModelPlusList[i].predictT(x)
            modelVotes[i, :, :] = pred.detach()

        finalVotes = torch.zeros((sampleSize, self.defense.ClassNum)).to(self.device)
        for i in range(sampleSize):
            currentTally = torch.zeros(self.defense.ClassNum).to(self.device)
            for j in range(self.defense.ModelNum):
                currentVote = modelVotes[j, i, :].argmax()
                currentTally[currentVote] += 1
            if currentTally.max() >= self.defense.Threshold:
                # Normal class selection
                finalVotes[i, currentTally.argmax()] = 1.0
            else:
                # Randomize if threshold fails, or just give to the adversarial noise class
                # Here we just output zero vectors so precision drops and the prediction is 'wrong' when evaluated via argmax,
                # but matching dimensions (only 2 classes) prevents mismatched index errors downstream.
                # We could set both to uniform probabilities for rejection
                random_class = torch.randint(0, self.defense.ClassNum, (1,)).item()
                finalVotes[i, random_class] = 1.0
        return finalVotes
