# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as pr:
    install_requires = pr.readlines()

setup(
    name="xai_evals",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="A package for model explainability and explainability comparision for tabular data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="aryaxai deep learning backtrace, ML observability",
    license="MIT",
    url="https://github.com/AryaXAI/xai_evals",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="."),
    python_requires=">=3.0",
    install_requires=install_requires,
    package_data={
        "": ["*.md", "*.txt"],
    },
)
