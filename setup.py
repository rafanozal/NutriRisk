from setuptools import setup, find_packages

setup(
    name        = "NutriRisk",
    version     = "0.9",
    package_dir = {"": "src"},                # Tell Python to look inside the "src" folder
    packages    = find_packages(where="src"), # Automatically find the packages inside the "src" folder
)