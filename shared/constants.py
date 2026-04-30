from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]  # ballot_black_box/
_CKPT = _ROOT / "models" / "checkpoints"
_PROTO_CKPT = _ROOT / "explainable" / "ballot_protopnet" / "saved_models"
_DATA = _ROOT / "data"
_ADV = _ROOT / "adv_samples" / "rays"

CHECKPOINTS = {
    # "resnet20_only_bubbles": str(
    #     _CKPT / "ModelResNet20-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # ),
    "resnet20_combined": str(
        _CKPT / "ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    ),
    # "vgg16_only_bubbles": str(_CKPT / "ModelVgg16-B.th"),
    "vgg16_combined": str(_CKPT / "ModelVgg16-C2.th"),
    # "cait_only_bubbles": str(
    #     _CKPT / "ModelCaiT-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # ),
    "cait_combined": str(
        _CKPT / "ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    ),
    # "svm_only_bubbles": [
    #     str(
    #         _CKPT
    #         / "sklearn_SVM_OnlyBubbles_v2_Grayscale_Run1"
    #         / "base_pytorch_svm_OnlyBubbles_v2.pth"
    #     ),
    #     str(
    #         _CKPT
    #         / "sklearn_SVM_OnlyBubbles_v2_Grayscale_Run1"
    #         / "multi_output_svm_OnlyBubbles_v2.pth"
    #     ),
    # ],
    "svm_combined": [
        str(
            _CKPT
            / "sklearn_SVM_Combined_v2_Grayscale_Run1"
            / "base_pytorch_svm_combined_v2.pth"
        ),
        str(
            _CKPT
            / "sklearn_SVM_Combined_v2_Grayscale_Run1"
            / "multi_output_svm_combined_v2.pth"
        ),
    ],
    "expv2_resnet20_combined": "/home/syed_ayon_uri_edu/code/ballot_black_box/explainable/v2_2026/ballot_protopnet/resnet/results/runs/checkpoints/dainty-sapphire-xoyrf/dainty-sapphire-xoyrf_best.pth",
    "expv2_vgg16_combined": "/home/syed_ayon_uri_edu/code/ballot_black_box/explainable/v2_2026/ballot_protopnet/vgg/results/runs/vgg16/zesty-pebble-kxyai_best.pth",
    "mamba": "/home/syed_ayon_uri_edu/code/ballot_black_box/mamba/result/best_mamba_model.pth",
}

VAL_DATASETS = {
    "OnlyBubbles Val": str(
        _DATA / "kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth"
    ),
    "Combined Val": str(_DATA / "kaleel_final_dataset_val_Combined_Grayscale.pth"),
}

TRAIN_DATASETS = {
    "OnlyBubbles Train": str(
        _DATA / "kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth"
    ),
    "Combined Train": str(_DATA / "kaleel_final_dataset_train_Combined_Grayscale.pth"),
}

DATASETS = {**VAL_DATASETS, **TRAIN_DATASETS}
UNET_PRINTER_DATASET = "./UNet/simulated_printer/experiment_results/datasets/sim_print_generated_val_combined_mse.pt"

EXPERIMENTS = {
    # "resnet20_only_bubbles": {
    #     "ckpt_path": CHECKPOINTS["resnet20_only_bubbles"],
    #     "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    # },
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": UNET_PRINTER_DATASET,
    },
    # "cait_only_bubbles": {
    #     "ckpt_path": CHECKPOINTS["cait_only_bubbles"],
    #     "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    # },
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": UNET_PRINTER_DATASET,
    },
    # "vgg16_only_bubbles": {
    #     "ckpt_path": CHECKPOINTS["vgg16_only_bubbles"],
    #     "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    # },
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": UNET_PRINTER_DATASET,
    },
    # "svm_only_bubbles": {
    #     "ckpt_path": CHECKPOINTS["svm_only_bubbles"],
    #     "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    # },
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": UNET_PRINTER_DATASET,
    },
    "expv2_vgg16_combined": {
        "ckpt_path": CHECKPOINTS["expv2_vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
    "expv2_resnet20_combined": {
        "ckpt_path": CHECKPOINTS["expv2_resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
    "mamba": {
        "ckpt_path": CHECKPOINTS["mamba"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
}

ADV_SAMPLES = {
    "resnet20": str(_ADV / "ResNet"),
    "cait": str(_ADV / "CaiT"),
    "vgg16": str(_ADV / "VGG"),
    "svm": str(_ADV / "MultiOutputSVM"),
    "expv2_resnet20_combined": str(_ADV / "expv2ResNetCombined"),
    "expv2_vgg16_combined": str(_ADV / "expv2VGG16Combined"),
}
