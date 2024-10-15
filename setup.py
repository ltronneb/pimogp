from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pimogp',
    version='0.0.1',
    packages=find_packages("."),
    install_requires=requirements,
    include_package_data=True,
    package_data={'pimogp/pimogp': ['data/*']},
    python_requires='>=3.8',
    url='',
    license='',
    author='leiv',
    author_email='',
    description=''
)
