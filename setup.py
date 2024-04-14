from setuptools import setup, find_packages

setup(
    name='pdfchat',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "llama_index",
    ],
    # Additional metadata about your package.
    author='Chinmay Shrivastava',
    author_email='cshrivastava99@gmail.com',
    description='#',
    long_description_content_type='text/markdown',
)