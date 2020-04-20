from setuptools import setup, find_packages

d = {}
exec(open("spikefeatures/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

pkg_name = "spikefeatures"

setup(
    name=pkg_name,
    version=version,
    author="Alessio Buccino, Thijs Ruikes",
    author_email="alessiop.buccino@gmail.com",
    description="Python toolkit for computing extracellular waveforms features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpikeInterface/spikefeatures",
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
)
