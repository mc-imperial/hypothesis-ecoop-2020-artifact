import os

import setuptools


def local_file(name):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), name))


SOURCE = local_file("src")
README = local_file("README.md")

setuptools.setup(
    name="reducer-evaluations",
    # Not actually published on pypi
    version="0.0.0",
    author="David R. MacIver",
    author_email="david@drmaciver.com",
    packages=setuptools.find_packages(SOURCE),
    package_dir={"": SOURCE},
    url=("https://github.com/DRMacIver/reducer-evaluations/"),
    license="GPL v3",
    description="Evaluation experiments for ",
    zip_safe=False,
    install_requires=[],
    entry_points={"console_scripts": ["reducereval=reducereval.__main__:main"]},
    long_description=open(README).read(),
)
