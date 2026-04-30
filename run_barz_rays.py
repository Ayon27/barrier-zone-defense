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
from attacks.rays_attack.AttackWrappersRayS import RaySAttack
from shared.Tee import run_with_logging
from barz_utils import load_data, setup_barz_defense, evaluate_clean
from barz_configs import VoterConfigManager


def evaluate_rays(barz, defense_adapter, device, loader, epsilons):
    for eps in epsilons:
        print(
            f"\n=== Evaluating RayS Attack at Epsilon {eps:.4f} ({int(eps*255)}/255) ==="
        )
        rays_adv_loader = RaySAttack(
            device, defense_adapter, epsMax=eps, queryLimit=10000, cleanLoader=loader
        )
        rays_acc = 1.0 - barz.evaluateAdversarialAttackSuccessRate(rays_adv_loader)
        print(
            f"[+] Robust accuracy under RayS Attack (eps={eps:.4f}): {rays_acc:.4f}\n"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Models & Defense
    # Available configs: "base", "with_svm", "ramp_ce", "ramp_ce_ensemble" "diffusion_linf_only",
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

    # 4. RayS Attack Evaluation
    epsilons = [4 / 255, 8 / 255, 16 / 255, 255 / 255]
    evaluate_rays(barz, defense_adapter, device, loader, epsilons)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    out_path = str(_ROOT / "barz" / "results" / "blackbox_barz_rays_results_eps.log")
    run_with_logging(main, out_path)
