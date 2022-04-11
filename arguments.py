import argparse

def parse_args(mode):
    parser = argparse.ArgumentParser(description="Shortcut removal")
    if mode == 'train':
        parser.add_argument(
            "--downstream",
            dest="downstream",
            action="store_false",
            default=False,
            help="If you want to train downstream model",
        )
        parser.add_argument(
            "--clean_data",
            dest="clean_data",
            action="store_false",
            default=False,
            help="If you want to train with clean data",
        )
        parser.add_argument(
            "--shortcut",
            default="arrow",
            choices=[
                "arrow",
                "chromatic"
            ],
            type=str,
            help="Choose the type of shortcut added: arrow or chromatic",
        )
        parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
        parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
        parser.add_argument("--lambda_term", default=1e-10, type=float, help="lambda_term for lens network")
        parser.add_argument(
            "--lens_usage",
            dest="lens_usage",
            action="store_false",
            default=False,
            help="Whether you want to use Lens network or not",
        )
        parser.add_argument(
            "--full_adversarial",
            dest="full_adversarial",
            action="store_false",
            default=False,
            help="Whether you want to use the full adversarial loss",
        )
    elif mode == 'eval':
        parser.add_argument(
            "--downstream",
            dest="downstream",
            action="store_false",
            default=False,
            help="If you want to evaluate downstream model",
        )
        parser.add_argument(
            "--lens_usage",
            dest="lens_usage",
            action="store_false",
            default=False,
            help="Whether you want to use Lens network or not",
        )
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size")
    parser.add_argument(
        "--output_dir",
        default="./checkpoints",
        type=str,
        help="Path to save model to",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Naming of model. For example, '001'",
        required=True
    )
    return parser.parse_args()