from setuptools import setup
from setuptools import find_packages


setup(name='Kaulos',
      version='0.1.0',
      description='A computational neuroscience/deep learning library running on Keras, built as a backend for Neuroballad. ',
      author='Kaulos Development Team',
      author_email='http://www.bionet.ee.columbia.edu/people',
      url='',
      download_url='',
      license='MIT',
      install_requires=['neurokernel', 'neurodriver', 'neuroballad'],
      packages=find_packages())
