from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    description="Retrieval-Augmented Generation Library",
    author="Julian Velandia",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "scikit-learn",
        "numpy"
    ],
    python_requires=">=3.7",
)
