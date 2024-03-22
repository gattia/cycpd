from setuptools import Extension, setup

# potential way to read in version # ?
# https://blogs.nopcode.org/brainstorm/2013-05-20-pragmatic-python-versioning-via-setuptools-and-git-tags/


def readme():
    with open("README.MD") as f:
        return f.read()


try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            Extension(
                name="cython_functions",
                sources=["cycpd/cython/cython_functions.pyx"],
                include_dirs=["cycpd/cython/"],
            )
        ]
    )

except ImportError:
    ext_modules = None

try:
    import numpy as np
except ImportError:
    raise Exception(
        "Numpy must be installed to build this pacakge.\n Install with `pip install .` or run `pip install -r requirements.txt` before building."
    )

setup(
    name="cycpd",
    version="0.26",
    description="Numpy + Cython Implementation of the Coherent Point Drift Algorithm",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gattia/cycpd",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        "Topic :: Scientific/Engineering",
    ],
    keywords="image processing, point cloud, registration, mesh, surface",
    author="Anthony Gatti",
    author_email="anthony@neuralseg.com",
    license="MIT",
    ext_modules=ext_modules,
    # include_dirs=[np.get_include()],
    packages=["cycpd"],
    # package_data={"cycpd.cython": ["cython_functions.pxd"]},
    setup_requires=["Cython>=0.29", "setuptools"],
    install_requires=["numpy"],
    zip_safe=False,
)
