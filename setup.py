from setuptools import find_packages, setup

REQUIRES_PYTHON = ">=3.7.0"
NAME = "binder"

install_requires = [
    "emoji==1.7.0",
    "fuzzywuzzy==0.18.0",
    "nltk==3.6.2",
    "sqlparse==0.4.2",
    "recognizers-text-suite==1.0.2a2",
    "records"
]

setup(
    name=NAME,
    version="0.0.1",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    install_requires=install_requires,
)
