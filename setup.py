import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="lwcc",
    version="0.0.6",
    author="Matija Teršek, Maša Kljun",
    author_email="matijatersek@protonmail.com",
    description="A LightWeight Crowd Counting library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tersekmatija/lwcc",
    packages=setuptools.find_packages(),
    scripts=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=["numpy>=1.14.0", "torch>=1.6.0", "gdown>=3.10.1", "torchvision>=0.7.0", "Pillow>=8.0.0"]
)