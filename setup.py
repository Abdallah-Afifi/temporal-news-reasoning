from setuptools import setup, find_packages

setup(
    name="temporal-news-reasoning",
    version="0.1.0",
    description="Efficient Temporal Reasoning for News Understanding",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "datasets>=2.16.0",
    ],
)
