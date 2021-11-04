# pyre-strict

from typing import Type, TypeVar


T = TypeVar("T")

_CONFIG_ATTR = "_config_entry"


def config_entry(fn: T) -> T:
    """Decorator used to mark an object as the config entry-point for a class.

    Below is an exameple usage for this decorator. Only a single method in a
    class should be annoted as the @config_entry.

        class MyObject:
            @config_entry
            @staticmethod
            def from_config(config: MyConf) -> 'MyObject':
                ...
    """
    setattr(fn, _CONFIG_ATTR, None)
    return fn


def get_class_config_method(klass: Type[T]) -> str:
    """
    Args:
        klass: The class definition.

    Raises:
        ValueError if the klass does not define a single such method or the
        defined method is invalid.

    Returns:
        The fully qualified name of the method.
    """
    class_name = get_class_name_str(klass)
    fns = [fn for name, fn in klass.__dict__.items() if hasattr(fn, _CONFIG_ATTR)]
    if len(fns) != 1:
        raise ValueError(
            f"{class_name} has no config entrypoint. Did you use @config_entry to annotate the method?"
        )
    fn = fns[0]
    if not isinstance(fn, staticmethod):
        raise ValueError(
            f"{class_name}.{fn.__name__} is not a standalone function. Did you forget @staticmethod?"
        )
    return f"{class_name}.{fn.__func__.__name__}"


def get_class_name_str(klass: Type[T]) -> str:
    """
    Args:
        klass: The class definition.

    Returns:
        The fully qualified name of the given class.
    """
    return ".".join([klass.__module__, klass.__name__])
