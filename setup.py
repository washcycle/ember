import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ember",
    version="0.0.1",
    author="Matt Landowski",
    author_email="matthew.landowski@gmail.com",
    description="Categorical feature embedding generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/washcycle/ember",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)