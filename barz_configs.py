import torch
import sys
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.ModelFactory import ModelFactory
from shared.constants import CHECKPOINTS


class VoterConfigManager:

    _CONFIGS = {
        "base": ["base"],
        "with_svm": ["base", "svm"],
        "ramp_ce": ["ramp"],
        "ramp_ce_ensemble": ["base", "ramp"],
        "diffusion_linf": ["diffusion_linf"],
        "diffusion_linf_ensemble": ["base", "diffusion_linf"],
        "diffusion_linf_ramp_ce": ["ramp", "diffusion_linf"],
        "diffusion_linf_ramp_ce_ensemble": [
            "base",
            "ramp",
            "diffusion_linf",
        ],
    }

    def __init__(self, device):
        self.device = device
        self.factory = ModelFactory(device)

    # Loads the 3 base vanilla models.
    def _get_base_voters(self):
        return {
            "resnet20": self.factory.get_model(
                "resnet20_combined", CHECKPOINTS["resnet20_combined"]
            ),
            "cait": self.factory.get_model(
                "cait_combined", CHECKPOINTS["cait_combined"]
            ),
            "vgg16": self.factory.get_model(
                "vgg16_combined", CHECKPOINTS["vgg16_combined"]
            ),
        }

    # Dynamically loads adversarial models mapping to distinct WRN modules.
    def _load_wrn_adv(self, ckpt_path, arch_dir, module_file, name, state_key=None):
        arch_path = _ROOT / arch_dir / module_file

        spec = importlib.util.spec_from_file_location("wrn_adv", arch_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["wrn_adv"] = module
        spec.loader.exec_module(module)

        model = module.wideresnetwithswish_voting(name).to(self.device)
        ckpt = torch.load(
            _ROOT / ckpt_path, map_location=self.device, weights_only=False
        )

        state = ckpt[state_key] if state_key else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    # Returns the voter models dict for the selected experiment config.
    def _load_ramp_ce(self):
        return self._load_wrn_adv(
            "adver-models/ramp-adver-models/apgd-linf-l1-ce/ep_20_0.pth",
            "adver-models/ramp-adver-models/model-architecture-file",
            "wideresnetwithReluRamp_voting.py",
            "wrn-28-10-relu",
        )

    def _load_diffusion_linf(self):
        return self._load_wrn_adv(
            "adver-models/diffusion-adver-models/diffusion-linf-pgd/linf-state-last.pt",
            "adver-models/diffusion-adver-models/model-archi-file",
            "wideresnetwithswish_voting.py",
            "wrn-28-10-swish",
            state_key="model_state_dict",
        )

    def _load_svm(self):
        return self.factory.get_model("svm_combined", CHECKPOINTS["svm_combined"])

    def _extra_voters(self):
        return {
            "svm": self._load_svm,
            "ramp": self._load_ramp_ce,
            "diffusion_linf": self._load_diffusion_linf,
        }

    def get_voters(self, config_type: str) -> dict:

        voters = {}

        for item in self._CONFIGS[config_type]:

            if item == "base":
                voters.update(self._get_base_voters())
            else:
                voters[item] = self._extra_voters()[item]()

        return voters
