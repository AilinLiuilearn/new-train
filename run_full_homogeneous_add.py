"""Full-only native-channel Add baseline; optional same-backbone fusion ablation.

Example: python run_full_homogeneous_add.py --encoder_pair mit_b1 ...
"""
from configs.full_homogeneous import FullHomogeneousConfig
from models.full_homogeneous import build_homogeneous_model
from run_full_structure_response_fusion import main as run_experiment


def main():
    run_experiment(config_class=FullHomogeneousConfig,
                   model_builder=build_homogeneous_model,
                   experiment_type='full_homogeneous')


if __name__ == '__main__':
    main()
