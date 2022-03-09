from setuptools import find_packages, setup
from os.path import basename, splitext
setup(
    name='DisasterResponse',
    version='1.0',
    author='Sang-Ho Kim',
    author_email='skim21@live.com', 
    packages=find_packages(),
    package_data={'': ['*.py'], 'app': ['*.py']},
)
