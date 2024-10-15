"""
PolyGraph installation setup
"""

import setuptools

VERSION = "1"

setuptools.setup(
    name="polygraphs_updated",
    version=VERSION,
    description="PolyGraphs",
    long_description="",
    author="Alexandros Koliousis",
    author_email="ak@akoliousis.com",
    url="https://github.com/Mar1aFede/polygraphs_updated",
    license="MIT",
    keywords="test test",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=["torch", "dgl", "notebook", "matplotlib", "pylint", "flake8", "PyYaml", "pandas", "h5py"],
    python_requires=">=3",
    package_data={'polygraphs': ['logging.yaml']},
    include_package_data=True,
)
