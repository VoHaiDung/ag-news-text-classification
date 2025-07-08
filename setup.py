from setuptools import setup, find_packages

setup(
    name="ag_news_classification",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
