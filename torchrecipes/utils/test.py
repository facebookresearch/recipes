from functools import wraps
from tempfile import TemporaryDirectory
from typing import TypeVar, Callable
from unittest import TestCase

from pyre_extensions import ParameterSpecification


TParams = ParameterSpecification("TParams")
TReturn = TypeVar("TReturn")


def tempdir(func: Callable[TParams, TReturn]) -> Callable[TParams, TReturn]:
    """A decorator for creating a tempory directory that is cleaned up after function execution."""

    @wraps(func)
    def wrapper(
        self: TestCase, *args: TParams.args, **kwargs: TParams.kwargs
    ) -> TReturn:
        with TemporaryDirectory() as temp:
            return func(self, temp, *args, **kwargs)

    return wrapper
