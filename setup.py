from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Olivier Dietrich",
    author_email="author@example.com",
    description="Detecting building destruction from satellite images time-series in Ukraine",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
