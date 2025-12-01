from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gcsopt",
    version="0.1.3",
    description="Library for solving optimization problems over Graphs of Convex Sets (GCS).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tobia Marcucci",
    author_email='marcucci@ucsb.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "cvxpy >=1.7.4",
        "pytest",
    ],
)
