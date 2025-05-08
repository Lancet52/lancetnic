from setuptools import setup, find_packages

# Читаем README.md в отдельной переменной
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lancetnic",         
    version="1.0.2",           
    author="Lancet52",
    author_email="lancetFPV@yandex.ru",
    description="A tool for working with text data",
    long_description=long_description,  
    long_description_content_type="text/markdown",
    url="https://github.com/Lancet52",
    packages=find_packages(),
    install_requires=[          
        "torch==2.5.1",           
        "torchaudio==2.5.1",
        "torchvision==0.20.1",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "matplotlib==3.10.1",
        "seaborn==0.13.2",        
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",   
)