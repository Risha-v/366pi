from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    return "EuroSAT Land Cover Classification Project"

# Read requirements
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="eurosat-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="EuroSAT Land Cover Classification using Deep Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eurosat-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "streamlit>=1.25.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "plotly>=5.0.0",
        "opencv-python>=4.5.0",
        "psutil>=5.8.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "export": ["onnx>=1.12.0", "onnxruntime>=1.12.0"],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "eurosat-train=train_model:main",
            "eurosat-predict=predict:main",
            "eurosat-prepare=prepare_data:main",
            "eurosat-export=export_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/eurosat-classifier/issues",
        "Source": "https://github.com/yourusername/eurosat-classifier",
        "Documentation": "https://eurosat-classifier.readthedocs.io/",
    },
)
