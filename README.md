# Iterative Prompt Optimization

A Python library for iterative prompt optimization using various LLM providers.

## Overview

This library provides a framework for optimizing prompts used in classification tasks with Large Language Models (LLMs). It supports multiple LLM providers and implements an iterative process to refine prompts based on performance metrics.

## Features

- Support for multiple LLM providers (OpenAI, Azure OpenAI, Anthropic, Google, Ollama)
- Iterative prompt optimization
- Evaluation metrics calculation
- Logging and analysis of results
- Dashboard generation for visualizing optimization progress

## Installation
```bash
pip install iterative-prompt-optimization
```

## Usage
```python
from iterative_prompt_optimization import optimize_prompt
import pandas as pd

# Prepare your evaluation data
eval_data = pd.DataFrame({
    'text': ['I am happy!', 'excited about the trip', 'I am sad!', ' i am not sure how i feel'],
    'label': [1, 1, 0, 1]
    })
# Define your initial prompt and output format
initial_prompt = "Classify the following text as either positive (1) or negative (0):"
output_format_prompt = "Output your classification as a single digit: 0 or 1"
#Define the output schema
output_schema = {
    'regex_pattern': r'^(\d)$',
    'value_mapping': {'0': 0, '1': 1}
    }
# Run the optimization process
best_prompt, best_metrics = optimize_prompt(
    initial_prompt=initial_prompt,
    output_format_prompt=output_format_prompt,
    eval_data=eval_data,
    iterations=5,
    output_schema=output_schema
)
print("Best Prompt:", best_prompt)
print("Best Metrics:", best_metrics)
```

```python
# Or passing all the parameters to the function:
# Define model providers and models for evaluation and optimization
eval_provider = "ollama"
eval_model = "llama3.1"
optim_provider = "ollama"
optim_model = "llama3.1"

best_prompt, best_metrics = optimize_prompt(
    initial_prompt=initial_prompt,
    output_format_prompt=output_format_prompt,
    eval_data=eval_data,
    iterations=5,
    output_schema=output_schema,
    eval_provider=eval_provider,
    eval_model=eval_model,
    eval_temperature=0.7,
    optim_provider=optim_provider,
    optim_model=optim_model,
    optim_temperature=0,
    use_cache=True  # Set to False if you want to disable caching
)
print("Best Prompt:", best_prompt)
print("Best Metrics:", best_metrics)
```

## Evaluation and Prompt Optimization Process

The evaluation and prompt optimization process follows these steps:

1. Initial Setup: Load the initial prompt and evaluation data.
2. Evaluation: Test the current prompt against the evaluation data.
3. Metrics Calculation: Calculate performance metrics (precision, recall, F1 score, etc.).
4. Analysis: Analyze false positives, false negatives, and invalid outputs.
5. Prompt Generation: Generate a new prompt based on the analysis.
6. Iteration: Repeat steps 2-5 for the specified number of iterations.
7. Results: Select the best-performing prompt based on F1 score.

```
        +-------------------+               
        | 1. Initial Setup  |               
        | (Initial prompt   |               
        |   & eval data)    |               
        +--------+----------+               
                |                            
                v                            
        +-------------------+               
        | 2. Evaluation     |<--------------+
        | (Test prompt on   |               |
        |    eval data)     |               |
        +--------+----------+               |
                |                           |
                v                           |
        +-------------------+               |
        | 3. Metrics        |               |
        |   Calculation     |               |
        +--------+----------+               |
                |                   +-------------------+
                |                   |    Iteration      |
                |                   | (Repeat process   |
                |                   |  for N iterations)|
                |                   +--------+----------+
                v                           |
        +-------------------+               |
        | 4. Analysis       |               |
        |  (FP, FN, TP,     |               |
        |  Formatting)      |               |
        +--------+----------+               |
                |                           |
                v                           |
        +-------------------+               |
        | 5. Prompt         |               |
        |    Generation     |               |
        | (Create new prompt|               |
        | based on analysis)|               |
        +--------+----------+               |
                |                           |
                +---------------------------+
                |
                v
        +-------------------+
        | 7. Results        |
        | (Select best      |
        | performing prompt)|
        +-------------------+
```


## Configuration

The library uses environment variables for API keys and other configuration settings. Create a `.env` file in your project root with the following structure:

    OPENAI_API_KEY=your_openai_api_key
    -
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key
    AZURE_OPENAI_API_BASE=your_azure_openai_api_base
    AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
    -
    OLLAMA_MODEL=your_preferred_ollama_model
    -
    ANTHROPIC_API_KEY=your_anthropic_api_key
    -
    GOOGLE_AI_API_KEY=your_google_ai_api_key


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Diagram
[Evaluation and Prompt Optimization Process]
