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
    author="Daniel Fiuza Dosil ",
    author_email="daniel@helpfirst.ai",
    description="A library for iterative prompt optimization using various LLM providers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HelpFirst/AI-Prompt-Optimiser",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)