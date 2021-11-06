from dataclasses import asdict
from typing import Any, Dict


# pyre-fixme[2]: Parameter must be annotated.
def conf_asdict(datacls_obj) -> Dict[str, Any]:
    """
    The dataclasses we provide may contain Hydra specific fields.
    Use this method instead of dataclasses.asdict to remove those.

    Args
        :param datacls_obj: a dataclass object
        :return: dict of the dataclass
    """
    args = asdict(datacls_obj)
    if "_target_" in args:
        del args["_target_"]
    return args
