from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='tfignite',
      version='0.0.4',
      description='Hight level training routine for tensorflow.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Shih-Ming Wang',
      author_email='swang150@ucsc.edu',
      url='https://github.com/ipod825/tfignite',
      download_url='https://github.com/ipod825/tfignite/tarball/0.0.1',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      install_requires=[],
      extras_require={},
      packages=find_packages())
