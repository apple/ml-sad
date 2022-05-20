import os

from setuptools import setup

_setup_dir = os.path.dirname(os.path.abspath(__file__))

VERSION_FILE = os.path.join(_setup_dir, "VERSION")
ENV_REQUIREMENTS_FILE = os.path.join(_setup_dir, "requirements-env.txt")


def read_version(file_path):
    """Read version from file path"""
    with open(file_path) as version_file:
        version = version_file.read().strip()
    return version


# uses env requirements (with freezed versions)
# for packaging for now
def read_pkg_requirements(file_path):
    """Read packaging requirements from requirements-env.txt"""
    with open(file_path) as f:
        required = f.read().splitlines()
    return required


setup(
    version=read_version(VERSION_FILE),
    install_requires=read_pkg_requirements(ENV_REQUIREMENTS_FILE),
)
