from setuptools import setup, Extension

def readme():
    with open('README.MD') as f:
        return f.read()

try:
    from Cython.Build import cythonize

    ext_modules = cythonize([
        Extension("cycpd.cython_functions", ["cycpd/cython/cython_functions.pyx"])
    ])

except ImportError:
    ext_modules = None

setup(name='cycpd',
      version='0.0.1',
      description='Numpy + Cython Implementation of the Coherent Point Drift Algorithm',
      long_description=readme(),
      url='',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
      ],
      keywords='image processing, point cloud, registration, mesh, surface',
      author='Anthony Gatti',
      author_email='anthony@neuralseg.com',
      license='MIT',
      ext_modules=ext_modules,
      packages=['cycpd'],
      install_requires=['numpy', 'future', 'Cython'],
      zip_safe=False)
