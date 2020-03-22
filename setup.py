from setuptools import setup
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))


def fpath(name):
    return os.path.join(os.path.dirname(__file__), name)


def read(fname):
    return codecs.open(fpath(fname), encoding='utf-8').read()


requirements = read(fpath('requirements.txt'))

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
with codecs.open(os.path.join(here, 'sk_dsp_comm', 'sdr_helper', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), about)

setup(name='sdr-helper',
      version=about['__version__'],
      description='SDR Helper',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Mark Wickert',
      author_email='mwickert@uccs.edu',
      maintainer='Chiranth Siddappa',
      maintainer_email='chiranthsiddappa@gmail.com',
      url='https://github.com/scikit-dsp-comm/sdr_helper',
      packages=['sk_dsp_comm.sdr_helper'],
      zip_safe=False,
      license='BSD',
      install_requires=requirements.split(),
      test_suite='nose.collector',
      tests_require=['nose','numpy', 'tox'],
      extras_require={
            'rtlsdr': ['pyrtlsdr']
      },
      python_requires = '>=3.5',
     )
