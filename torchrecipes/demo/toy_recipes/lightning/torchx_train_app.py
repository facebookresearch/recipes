import torchx.specs as specs


def lightning_train(image: str = "lightning_train_app:latest", *args: str) -> specs.AppDef:
    return specs.AppDef(
        name="toy_recipe_lightning_train_app",
        roles=[
            specs.Role(
                name="lightning_train_app",
                image=image,
                entrypoint="python",
                args=[
                    "-m", "lightning_train_app",
                ] + [v for arg in args for v in arg.split()],
            )
        ],
    )
