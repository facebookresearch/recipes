import torchx.specs as specs


def train(image: str = "train:latest", *args: str) -> specs.AppDef:
    return specs.AppDef(
        name="text_classification_fine_tune_xlmr",
        roles=[
            specs.Role(
                name="trainer",
                image=image,
                entrypoint="python",
                args=[
                    "-m", "train",
                ] + [v for arg in args for v in arg.split()],
            )
        ],
    )