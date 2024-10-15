"""
PolyGraph installation setup
"""

import setuptools

VERSION = "2"

setuptools.setup(
    name="polygraphs_updated",  # This is the name of the package as it will be installed
    version=VERSION,
    description="PolyGraphs",
    long_description="",
    author="Alexandros Koliousis",
    author_email="ak@akoliousis.com",
    url="https://github.com/Mar1aFede/polygraphs_updated",
    license="MIT",
    keywords="test test",
    packages=['polygraphs_updated'],  # Reference the folder containing the Python code
    install_requires=[
        "torch", "dgl", "notebook", "matplotlib", "pylint", "flake8", "PyYaml", "pandas", "h5py"
    ],
    python_requires=">=3",
    package_data={'polygraphs_updated': ['logging.yaml']},
    include_package_data=True,
)
