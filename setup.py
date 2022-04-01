# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import glob
import os
import re
import shutil
import subprocess
from typing import List

from setuptools import setup, find_packages
import distutils.command.clean  # isort:skip

_PACKAGE_NAME: str = "torchrecipes"
_VERSION_FILE: str = "version.py"
_README: str = "README.md"
_REQUIREMENTS: str = "requirements.txt"
_DEV_REQUIREMENTS: str = "dev-requirements.txt"
_GITIGNORE: str = ".gitignore"


def get_version() -> str:
    """Retrieves the version of the library."""
    if version := os.getenv("BUILD_VERSION"):
        return version
    cwd = os.path.dirname(os.path.abspath(__file__))
    version_file_path = os.path.join(_PACKAGE_NAME, _VERSION_FILE)
    version_regex = r"__version__: str = ['\"]([^'\"]*)['\"]"
    with open(version_file_path, "r") as f:
        search = re.search(version_regex, f.read(), re.M)
    assert search
    version = search.group(1)

    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
        version += "+" + sha[:7]
    except Exception:
        pass
    return version


def get_long_description() -> str:
    """Fetch project description as Markdown."""
    with open(_README, mode="r") as f:
        return f.read()


def get_requirements() -> List[str]:
    """Fetch requirements."""
    with open(_REQUIREMENTS, mode="r") as f:
        return f.readlines()


def get_dev_requirements() -> List[str]:
    """Fetch requirements for library development."""
    with open(_DEV_REQUIREMENTS, mode="r") as f:
        return f.readlines()


class clean(distutils.command.clean.clean):
    def run(self) -> None:
        with open(_GITIGNORE, "r") as f:
            ignores = f.readlines()
            for wildcard in filter(None, ignores):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


if __name__ == "__main__":
    version: str = get_version()
    print("Building wheel {}-{}".format(_PACKAGE_NAME, version))

    setup(
        # Metadata
        name=_PACKAGE_NAME,
        version=version,
        author="PyTorch Ecosystem Foundations Team",
        author_email="luispe@fb.com",
        description="Prototype of training recipes for PyTorch",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/recipes",
        license="BSD-3",
        keywords=["pytorch", "machine learning"],
        python_requires=">=3.8",
        install_requires=get_requirements(),
        include_package_data=True,
        # Package info
        packages=find_packages(),
        # pyre-fixme[6]: Expected `Mapping[str, typing.Type[setuptools.Command]]`
        #  for 15th param but got `Mapping[str, typing.Type[clean]]`.
        cmdclass={
            "clean": clean,
        },
        extras_require={"dev": get_dev_requirements()},
        # PyPI package information.
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
