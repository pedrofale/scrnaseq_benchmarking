from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="benchmark",
    version="0.1",
    author="Pedro F. Ferreira",
    description="Package for comparing scRNA-seq models.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
