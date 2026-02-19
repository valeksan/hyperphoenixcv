from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hyperphoenixcv",
    version="0.1.0",
    author="valeksan",
    author_email="vitaljax001@gmail.com",
    description="Hyper-parameter search that rises from the ashes of interrupted experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/valeksan/hyperphoenixcv"
)
