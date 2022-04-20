from setuptools import setup, find_packages

with open("README.md", "r") as description_file:
    long_description = description_file.read()

setup(
        name="ares-pytorch",
        version="0.1.0",
        description="An adversarial attacks benchmark platform for robustness of image-based models.",
        long_description=long_description,
        author="Carles V, Hang S",
        url="https://github.com/vcarles/benchmark",
        python_requires=">=3.9",
        packages_dir={"": "src"},
        packages=find_packages(where="src")
    )
