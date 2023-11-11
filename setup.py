from setuptools import find_packages, setup
from os import path


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "torchtops", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [line.strip() for line in init_py if line.startswith("__version__")][
        0
    ]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


setup(
    name="torchtops",
    version=get_version(),
    author="yokosyun",
    license="MIT",
    author_email="yoko.syun@gmail.com",
    url="https://github.com/yokosyun/TorchTOPS",
    description="TOPS profiler of each layers of model",
    python_requires=">=3.6",
    install_requires=[
        "fvcore",
        "numpy",
        "torch",
    ],
    packages=find_packages(exclude=("*test*",)),
)
