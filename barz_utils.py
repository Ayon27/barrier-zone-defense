import torch
from torch.utils.data import DataLoader, TensorDataset

from models.ModelFactory import ModelFactory
from shared.constants import CHECKPOINTS
from ModelPlus import ModelPlus
from BarrierZoneDefense import BarrierZoneDefense
from attacks.adba_attack.adba_attack import ADBA_AttackWrapper
from shared.utils import GetCorrectlyIdentifiedSamplesBalanced
from DefenseWrapper import DefenseWrapper


def load_data(path, device, num_samples, model, batch_size=512):
    data = torch.load(path, map_location=device, weights_only=False)
    dataset = TensorDataset(data["data"].float(), data["binary_labels"].long())

    # load everything into a loader first so GetCorrectlyIdentifiedSamplesBalanced can sample
    raw_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if model:
        return GetCorrectlyIdentifiedSamplesBalanced(
            model,
            totalSamplesRequired=num_samples,
            dataLoader=raw_loader,
            numClasses=2,
            device=device,
        )
    raise ValueError("Model must be provided")


def setup_barz_defense(models_dict, device, batch_size=512):
    model_plus_list = []
    for name, model in models_dict.items():
        mp = ModelPlus(name, model, device, 40, 50, batch_size)
        model_plus_list.append(mp)

    # Dynamic threshold based on number of loaded models (100% agreement like before where it was 4/4)
    # If using removing SVM, there will be 3 models. So threshold will be 3.
    barz = BarrierZoneDefense(model_plus_list, classNum=2, threshold=len(models_dict))
    defense_adapter = DefenseWrapper(barz, device)
    defense_adapter.eval()
    return barz, defense_adapter


def evaluate_clean(barz, loader):
    clean_acc = barz.validateD(loader)
    print(f"\n[+] Clean BARZ Accuracy: {clean_acc:.4f}\n")
    return clean_acc
