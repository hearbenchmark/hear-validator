#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hearvalidator",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Submission Validator",
    author="HEAR 2021 NeurIPS Competition Committee",
    author_email="deep@neuralaudio.ai",
    url="https://github.com/neuralaudio/hear-validator",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/neuralaudio/hear-validator/issues",
        "Source Code": "https://github.com/neuralaudio/hear-validator",
    },
    packages=find_packages(),
    python_requires=">=3.6, <3.9",
    entry_points={
        "console_scripts": ["hear-validator=hearvalidator.validate:main"],
    },
    install_requires=["tensorflow>=2.0", "torch>=1.7"],
    extras_require={
        # Developer requirements
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
        ],
    },
)
