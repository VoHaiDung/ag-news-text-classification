from setuptools import setup, find_packages

setup(
    name="ag-news-text-classification",
    version="0.1.0",
    author="VoHaiDung",
    description="AG News Text Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "colab": ["google-colab"],
    }
)
