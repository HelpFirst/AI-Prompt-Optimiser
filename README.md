# Iterative Prompt Optimization

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



This repository hosts the Iterative Prompt Optimization Framework, an open-source Python library developed by Daniel Fiuza, from the HelpFirst.ai team. It automates the process of refining prompts to improve text classification performance with Large Language Models (LLMs), without requiring access to the model's internal parameters or fine-tuning.

> **Note**  
> - **Requirements**: this framework only work for binary and multiclass classification tasks. Currently only supports JSON output-like format with a chain of thought output to capture model reasoning steps.

> **Note**  
> This library originated from internal R&D efforts at HelpFirst.ai to improve our LLM-based classification systems. While we've found it valuable in our work and are excited to share it with the community, please note:
> - This is actively used but still evolving
> - Some edge cases may not be fully handled
> - Documentation and codebase are being improved
> 
> **We welcome contributions!** See [Contributing] section below.

---
## 🔍 Introduction

### Why Prompt Engineering for Text Classification?
Traditional machine learning (ML) classification models often require extensive data preparation, feature engineering, and training cycles. Large Language Models, on the other hand, can quickly adapt to various tasks through prompt engineering, where instructions are carefully structured to guide the model's output.

However, prompt engineering can be time-consuming and relies heavily on trial and error. To address this challenge, we introduce an automated iterative approach to refine prompts systematically.

### ⚙️ Core API: optimize_prompt()

```python
def optimize_prompt(
    initial_prompt: str,               # Starting prompt template
    eval_data: pd.DataFrame,           # Must contain 'text' and 'label' columns
    output_schema: dict,               # Requires chain_of_thought_key and classification_key
    iterations: int = 3,               # Number of optimization cycles (5-15 recommended)
    eval_provider: str = None,         # Provider for evaluation model
    eval_model: str = None,            # Model name for evaluation
    optim_provider: str = None,        # Provider for optimization model
    optim_model: str = None,           # Model name for prompt generation
    problem_type: str = 'binary',      # 'binary' or 'multiclass'
    output_format_prompt: str = None,  # Instructions for output formatting
    # ... other optional parameters ...
) -> tuple[str, dict]:  # Returns (optimized_prompt, final_metrics)
```
[TODO: add a link here to the examples:
- binary: /Users/danielfiuzadosil/Documents/GitHub/AI-Prompt-Optimiser/examples/binary_classifier/example_sentiments.ipynb
- multiclass: /Users/danielfiuzadosil/Documents/GitHub/AI-Prompt-Optimiser/examples/muticlass_classifier/example_news.ipynb]

### ⚙️ How Does It Work? (Teacher-Student Framework)

This framework improves LLM classification through iterative prompt refinement, mimicking how educators enhance exam instructions:

**1. Student's Initial Exam Attempt**  
*(Handled by `evaluation.evaluate_prompt()`)*  
The student model (evaluation LLM) executes the current prompt:
- **Input**: Prompt (exam instructions) + Text samples (test questions)
- **Output**: Predictions with chain-of-thought reasoning
- **Error Types**:
  - 🚫 False Positives: Incorrect positive classifications
  - 🚫 False Negatives: Missed true positives
  - ❌ Invalid Outputs: Formatting mistakes

**2. Teacher's Error Diagnosis**  
*(Implemented in `prompt_generation*.py`)*  
The teacher model (optimization LLM) analyzes errors:
```python
# Core analysis pattern
analysis = {
    'true_positives': analyze_success_patterns(true_positives),
    'false_positives': find_overly_broad_criteria(false_positives),
    'false_negatives': identify_missed_patterns(false_negatives),
    'invalid_outputs': detect_format_ambiguities(invalid_outputs)
}
```
- **True Positives**: Reinforces effective reasoning strategies
- **False Positives**: Tightens classification boundaries
- **False Negatives**: Clarifies edge case handling
- **Invalid Outputs**: Strengthens format specifications

**3. Instruction Refinement**  
*(Optimization loop in `optimize.py`)*  
The teacher updates the exam guidelines:

- Preserves successful elements
- Adds targeted error examples
- Specifies output constraints
- Clarifies ambiguous instructions

**4. Quality Assurance Check** *(Optional)*  
*(Via `VALIDATION_AND_IMPROVEMENT_PROMPT`)*  

Ensures new prompts:
1. Maintain task focus and clarity
2. Follow format requirements
3. Avoid unintended biases
4. Balance conciseness/completeness

**Outcome**:
- ✅ **Valid**: Proceeds to student retesting  
- 🛠 **Needs Improvement**: Further refines prompt, and provides an improved prompt. 

**5. Retesting & Iteration**  
*(Main loop in `optimize_prompt()`)*  
The student retakes the exam with refined instructions:

**Cycle Continues Until**:
- Target metrics achieved, or
- Max iterations reached

**Why This Works**  
With fixed model weights, better prompts enable:
- 🎯 **Precision Boost**: Error examples focus model attention
- 🧠 **Reasoning Structure**: Chain-of-thought reduces guessing
- 📝 **Format Compliance**: Schema validation minimizes parsing errors
- 🔁 **Continuous Improvement**: Multiple refinement passes

**Implementation Notes**  
- Supports binary & multiclass workflows
- Customizable validation strictness
- Teacher's Diagnosis prompt instructions can be customized

## ✨ Features

- Support for multiple LLM providers (OpenAI, Azure OpenAI, Anthropic, Google, Ollama)
- Iterative prompt optimization
- Evaluation metrics calculation (precision, recall, f1, accuracy, etc.)
- Logging and analysis of results
- Dashboard generation for visualizing optimization progress
- Automates the refinement of prompts based on misclassifications (false positives, false negatives, etc.) and invalid outputs

---

## 📦 Installation

To install the package:
```bash
pip install iterative-prompt-optimization
```

**Requirements**:
- Python 3.8+
- Create a `.env` file with your API keys:

```env
# .env.example
OPENAI_API_KEY="your-key-here"
ANTHROPIC_API_KEY="your-key-here" 
AZURE_OPENAI_API_KEY="your-key-here"
AZURE_OPENAI_ENDPOINT="your-endpoint-here"
GOOGLE_API_KEY="your-key-here"
```

---

## 🚀 Quick Start

### Binary Classification Example
```python
from iterative_prompt_optimization import optimize_prompt
import pandas as pd

# Sample binary evaluation data
binary_data = pd.DataFrame({
    'text': [
        'Loved the cinematography despite weak plot',
        'Terrible acting ruined a great concept',
        'Masterful storytelling with brilliant performances'
    ],
    'label': [1, 0, 1]  # 1=positive, 0=negative
})

# Binary output schema
binary_schema = {
    'chain_of_thought_key': 'reasoning',
    'classification_key': 'sentiment',
    'classification_mapping': {1: "positive", 0: "negative"}
}

# Initial binary prompt
binary_prompt = """Analyze movie review sentiment. Output JSON with:
- "reasoning": your step-by-step analysis
- "sentiment": 1 (positive) or 0 (negative)"""

# Run optimization
binary_best_prompt, binary_metrics = optimize_prompt(
    initial_prompt=binary_prompt,
    eval_data=binary_data,
    output_schema=binary_schema,
    iterations=3
)

### Multiclass Classification Example
```python
# Sample multiclass evaluation data
multiclass_data = pd.DataFrame({
    'text': [
        'The product works well but delivery was late',
        'Complete waste of money',
        'Average experience, nothing special'
    ],
    'label': [2, 0, 1]  # 0=negative, 1=neutral, 2=positive
})

# Multiclass output schema
multiclass_schema = {
    'chain_of_thought_key': 'analysis',
    'classification_key': 'rating',
    'classification_mapping': {
        0: "negative",
        1: "neutral", 
        2: "positive"
    }
}

# Initial multiclass prompt
multiclass_prompt = """Classify customer feedback into categories. Output JSON with:
- "analysis": detailed reasoning
- "rating": 0 (negative), 1 (neutral), or 2 (positive)"""

# Run optimization
mc_best_prompt, mc_metrics = optimize_prompt(
    initial_prompt=multiclass_prompt,
    eval_data=multiclass_data,
    output_schema=multiclass_schema,
    iterations=3,
    problem_type='multiclass'
)

print(f"Optimized multiclass prompt:\n{mc_best_prompt}")
```

**Key Requirements for Both Examples**:
- `output_schema` must contain:
  - `chain_of_thought_key`: Key for model's reasoning steps
  - `classification_key`: Key for final classification
  - `classification_mapping`: Dictionary mapping numeric labels to class names
- Prompts must explicitly request JSON output format
- Labels must be integers (0-N for multiclass)
- Chain of thought must be included in output

These examples will generate optimized prompts and interactive dashboards showing the optimization progress.

---

## 🔧 Configuration

Configure providers via environment variables or code:

```python
from iterative_prompt_optimization import config
import os

# Example: Azure OpenAI setup
config.set_model(
    provider="azure_openai",
    model="gpt-4",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

**Supported Providers**:
- OpenAI
- Azure OpenAI
- Anthropic
- Google AI
- Ollama (local models)

---

## 📊 Dashboard Preview

Generated through:
```python
# dashboard_generator.py
generate_iteration_dashboard(
    log_dir, 
    iteration, 
    results,  # From evaluation.py
    current_prompt,
    output_format_prompt,
    initial_prompt
)
```
Uses Jinja2 templates from dashboard_templates.py

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

1. :bug: **Report Bugs**  
   Open an issue with reproduction steps and environment details

2. :bulb: **Suggest Features**  
   Use the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md)

3. :computer: **Code Contributions**  
   ```bash
   # Development setup
   git clone https://github.com/your-repo/iterative-prompt-optimization.git
   cd iterative-prompt-optimization
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   pytest tests/ -v
   ```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🧭 Roadmap

- [x] Core optimization workflow (optimize.py)
- [x] Multi-provider support (model_interface.py)
- [ ] Automated hyperparameter tuning
- [x] Prompt version history (logging in utils.py)
- [ ] Interactive optimization dashboard

[View full roadmap](ROADMAP.md)

---

## 📧 Contact

For support and questions:
- [Open a Discussion](https://github.com/your-repo/discussions)
- Email: team@helpfirst.ai

**Security Note**: Never share API keys or sensitive information in public channels.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
