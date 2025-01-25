from setuptools import setup, find_packages

setup(
    name="SimpleRAGHuggingFace",
    version="0.1.0",
    description="A simple Retrieval-Augmented Generation (RAG) library using Hugging Face datasets.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Julian Camilo Velandia",
    author_email="velandiagutierrez@gmail.com",
    url="https://github.com/julianVelandia/SimpleRAGHuggingFace",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
