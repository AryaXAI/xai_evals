# setup.py
from setuptools import setup, find_packages

setup(
    name="xai_evals",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'dl_backtrace',
        'shap==0.46.0',
        'lime==0.2.0.1',
        'xgboost==2.1.3',
        'scikit-learn==1.3.2',
        'torch',
        'pandas==2.1.4',
        'numpy==1.26.4',
        'catboost==1.2.7',
        'lightgbm==4.5.0',
        'tensorflow==2.14.0',
        'captum==0.7.0',
        'tf-explain',
        'quantus'

    ],
    description="A package for model explainability and explainability comparision for tabular data",
    author="Pratinav Seth",
    author_email="pratinav.seth@arya.ai",
    url="https://github.com/AryaXAI/xai_evals",
)
