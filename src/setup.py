from setuptools import find_packages, setup

setup(
    name="xview3-d2",
    version="1.0",
    description="Detectron2 for xView3 Challenge",
    packages=find_packages(include=("xview3_d2", "xview3_d2.*")),
    python_requires=">=3.6.0",
)
