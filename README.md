# Iterative Prompt Optimization

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**  
> This library originated from internal R&D efforts at HelpFirst.ai to improve our LLM-based classification systems. While we've found it valuable in our work and are excited to share it with the community, please note:
> - This is actively used but still evolving
> - Some edge cases may not be fully handled
> - Documentation is being improved
> 
> **We welcome contributions!** See [Contributing](#contributing) below.

This repository hosts the Iterative Prompt Optimization Framework, an open-source Python library developed by Daniel Fiuza, from the HelpFirst.ai team. It automates the process of refining prompts to improve text classification performance with Large Language Models (LLMs), without requiring access to the model's internal parameters or fine-tuning.

---
## 🔍 Introduction

### Why Prompt Engineering for Text Classification?
Traditional machine learning (ML) classification models often require extensive data preparation, feature engineering, and training cycles. Large Language Models, on the other hand, can quickly adapt to various tasks through prompt engineering, where instructions are carefully structured to guide the model's output.

However, prompt engineering can be time-consuming and relies heavily on trial and error. To address this challenge, we introduce an automated iterative approach to refine prompts systematically.

### ⚙️ How Does It Work?
Our framework implements a sophisticated meta-prompting approach with reflection capabilities. Here's the detailed workflow:

1. **Dual LLM Setup**  
   - **Evaluator Model**: Tests the current prompt against your dataset
   - **Optimizer Model**: Analyzes results and generates improved prompts

2. **Evaluation Phase**  
   The evaluator LLM processes each text sample using the current prompt, while we:
   - Track valid/invalid outputs using regex pattern matching
   - Calculate precision, recall, accuracy, and F1 score
   - Identify misclassification patterns (FP/FN) and successful cases (TP)

3. **Meta-Prompt Construction**  
   We create a rich context for the optimizer LLM containing:
   ```python
   {
       "current_prompt": "Your classification instructions...",
       "performance_metrics": {precision: 0.85, recall: 0.78, ...},
       "misclassification_examples": [
           {
               "text": "sample text",
               "chain_of_thought": "model's reasoning steps",
               "true_label": 1,
               "predicted_label": 0
           }
       ],
       "validation_errors": ["missing formatting", "invalid syntax"]
   }
   ```

4. **Prompt Optimization Cycle**  
   The optimizer LLM uses special tools to refine prompts:
   - `analyze_patterns()`: Identifies systematic errors in classifications
   - `suggest_clarifications()`: Proposes instruction improvements
   - `validate_formatting()`: Ensures output schema compliance
   - `reflect()`: Internal "thinking" process to critique previous attempts

5. **Validation & Reflection**  
   Before finalizing new prompts, we:
   - Check adherence to prompt engineering best practices
   - Verify compatibility with output format requirements
   - Maintain version history of prompt iterations

6. **Iterative Refinement**  
   This process repeats, with each iteration:
   - Testing the most promising prompt variations
   - Focusing improvements on previous failure modes
   - Maintaining successful classification patterns

---

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

### Binary Classification
```python
from iterative_prompt_optimization import optimize_prompt
import pandas as pd

# Data format requirements:
# - DataFrame must contain 'text' and 'label' columns
# - Labels should be integers (0/1 for binary)
eval_data = pd.DataFrame({
    'text': ['I love this!', 'Terrible experience', 'Neutral comment'],
    'label': [1, 0, 0]
})

initial_prompt = "Classify sentiment as positive (1) or negative (0):"
output_format = "Output only 0 or 1 without explanation"

best_prompt, metrics = optimize_prompt(
    initial_prompt=initial_prompt,
    output_format_prompt=output_format,
    eval_data=eval_data,
    iterations=3,
    output_schema={
        'regex_pattern': r'^[01]$',
        'value_mapping': {'0': 0, '1': 1}
    }
)

print(f"Optimized Prompt: {best_prompt}")
print(f"Final Metrics: {metrics}")
```

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

![Optimization Dashboard](https://via.placeholder.com/800x400.png?text=Optimization+Dashboard+Preview)

View iteration history, metrics trends, and prompt evolution.

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

- [x] Core optimization workflow
- [x] Multi-provider support
- [ ] Automated hyperparameter tuning
- [ ] Prompt version diff visualization
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
