from setuptools import setup, find_packages

setup(
    name="cute-snippets",
    version="0.1.0",
    description="A Curated Assortment of CuTe DSL Experiments & Tutorials",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Add your dependencies here if needed
        "torch",
        "nvidia-cutlass-dsl",
    ],
)
