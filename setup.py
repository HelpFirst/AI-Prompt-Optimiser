from setuptools import setup, find_packages

setup(
    name="iterative_prompt_optimization",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "ollama",
        "openai",
        "anthropic",
        "google-generativeai",
        "scikit-learn",
        "tiktoken",
        "rich",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for iterative prompt optimization using various LLM providers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/iterative_prompt_optimization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)