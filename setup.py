from setuptools import setup, find_packages

setup(
    name='lomind',
    version='0.1.0',
    author='Dhaval',
    author_email="gametidhaval980@gmail.com",
    description='A simple, custom deep learning framework using NumPy.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/custom_deep_learning_framework',  # GitHub URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    
)
