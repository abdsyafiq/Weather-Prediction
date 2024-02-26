from setuptools import setup, find_packages

setup(
    name="logger",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "logger = logger.logger:log",
        ],
    },
    install_requires=[],
)
