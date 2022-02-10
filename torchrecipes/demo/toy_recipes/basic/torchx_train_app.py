import torchx.specs as specs


def train(image: str = "train_app:latest", *args: str) -> specs.AppDef:
    return specs.AppDef(
        name="toy_recipe_train_app",
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