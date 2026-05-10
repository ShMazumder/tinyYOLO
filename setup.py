from setuptools import setup, find_packages

setup(
    name="tinyYOLO",
    version="0.1.0",
    description="A modular, research-grade tiny object detection framework built on Ultralytics",
    author="TinyYOLO Research",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
)
