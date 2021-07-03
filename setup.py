#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hearkit",
    version="2021.0.1",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Starter Kit",
    author="HEAR 2021 NeurIPS Competition Committee",
    author_email="deep-at-neuralaudio.ai",
    url="https://github.com/neuralaudio/hear-starter-kit",
    download_url="https://github.com/neuralaudio/hear-starter-kit",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/neuralaudio/hear-starter-kit/issues",
        "Source Code": "https://github.com/neuralaudio/hear-starter-kit",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.6",
    install_requires=["librosa", "tensorflow", "torch"],
    extras_require={
        # Developer requirements
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
        ],
    },
    classifiers=[],
)
