from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperphoenixcv",
    version="0.2.0",
    author="valeksan",
    author_email="vitaljax001@gmail.com",
    description="Hyper-parameter search that rises from the ashes of interrupted experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valeksan/hyperphoenixcv",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.0.0",
        "joblib>=1.0.0",
    ],
    keywords="hyperparameter tuning, grid search, checkpointing, machine learning, scikit-learn",
    project_urls={
        "Bug Tracker": "https://github.com/valeksan/hyperphoenixcv/issues",
        "Source Code": "https://github.com/valeksan/hyperphoenixcv",
    },
)
