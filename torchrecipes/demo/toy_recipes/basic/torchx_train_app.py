import torchx.specs as specs


def train(image: str = "train_app:latest", *args: str) -> specs.AppDef:
    """
    Specify the entry point for torchx

    Example CLI:
        $ torchx run --scheduler local_cwd torchx_train_app.py:train "--num_epochs 10"
    """
    return specs.AppDef(
        name="toy_recipe_basic_train_app",
        roles=[
            specs.Role(
                name="train_app",
                image=image,
                entrypoint="python",
                args=[
                    "-m", "train_app",
                ] + [v for arg in args for v in arg.split()],
            )
        ],
    )