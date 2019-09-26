from setuptools import setup, find_packages


setup(name='dpp',
      version='0.1.0',
      description='Deep learning for temporal point processes',
      author='Oleksandr Shchur, Marin Bilos, Stephan Guennemann',
      author_email='shchur@in.tum.de',
      packages=find_packages('.'),
      zip_safe=False)
