import torch
import torch.nn as nn
import sys
from pathlib import Path

# __file__ is in barz/
_BARZ_DIR = Path(__file__).resolve().parent
_ROOT = _BARZ_DIR.parent

# Adjust paths to import local modules and cloned BARZ repository
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_BARZ_DIR))
sys.path.insert(0, str(_BARZ_DIR / "BarrierZoneTrainer" / "BarrierZoneTrainer"))

from shared.constants import VAL_DATASETS
from attacks.square_attack.square_attack_linf import SquareAttackLinf_Wrapper
from shared.Tee import run_with_logging
from barz_utils import load_data, setup_barz_defense, evaluate_clean
from barz_configs import VoterConfigManager


def evaluate_square(barz, defense_adapter, device, loader, epsilons):
    for eps in epsilons:
        print(
            f"\n=== Evaluating Square Attack (Linf) at Epsilon {eps:.4f} ({int(eps*255)}/255) ==="
        )
        adv_params = {
            "model": defense_adapter,
            "device": device,
            "dataLoader": loader,
            "eps": eps,
            "n_iters": 10000,
            "p_init": 0.05,
            "n_classes": 2,
            "targeted": False,
        }
        adv_loader = SquareAttackLinf_Wrapper(**adv_params)

        square_acc = 1.0 - barz.evaluateAdversarialAttackSuccessRate(adv_loader)
        print(
            f"[+] Robust accuracy under Square Attack (eps={eps:.4f}): {square_acc:.4f}\n"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Models & Defense
    # Available configs: "base", "with_svm", "ramp_ce", "ramp_ce_ensemble" "diffusion_linf",
    # "diffusion_linf_ensemble", "diffusion_linf_ramp_ce", "diffusion_linf_ramp_ce_ensemble"
    config_name = "base"
    print(f"Loading Voter Config: {config_name}")
    models_dict = VoterConfigManager(device).get_voters(config_name)

    barz, defense_adapter = setup_barz_defense(models_dict, device)

    # 2. Load Dataset
    loader = load_data(
        VAL_DATASETS["OnlyBubbles Val"], device, num_samples=500, model=defense_adapter
    )

    # 3. Clean Validation
    evaluate_clean(barz, loader)

    # 4. Square Attack Evaluation
    epsilons = [8 / 255]
    evaluate_square(barz, defense_adapter, device, loader, epsilons)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    out_path = str(
        _ROOT / "barz" / "results" / "new_config" / "base_barz_square_results_eps_8.log"
    )
    run_with_logging(main, out_path)
