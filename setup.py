# setup.py
from setuptools import setup, find_packages

setup(
    name="xai_evals",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'shap',
        'lime',
        'xgboost',
        'scikit-learn',
        'torch',
        'pandas',
        'numpy',
        'catboost',
        'lightgbm',
        'tensorflow==2.14.0',
        'captum'
    ],
    description="A package for model explainability and explainability comparision for tabular data",
    author="Pratinav Seth",
    author_email="pratinav.seth@arya.ai",
    url="https://github.com/AryaXAI/xai_evals",
)
