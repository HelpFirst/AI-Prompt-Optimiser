import pandas as pd
import sys
import os

# Add the parent directory to sys.path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.iterative_prompt_optimization import optimize_prompt

# Define initial prompt and output format
initial_prompt = (
    "Please look for the following risk factor: Is the client at risk of self-harm? For instance, do they mention suicidal thoughts or ideation? Do they imply they might do physical damage to themselves or to property? Do they reference wanting to 'end it all or say it's 'not worth living'? Please output: 1. risk_type: // suicidality, 2. risk_output: // 'risk present' : this means there is evidence this risk is present in the case 'risk not present' : there is evidence the risk is NOT present or there is no evidence whether the case contains that risk or not. (If in doubt, it is better to err on the side of caution and say 'risk present') 3. explanation: // State words/terms that indicate the reason the risk_output was chosen. Be brief in your explanation. State facts found in the text, do not infer. E.g. 'Client expressed suicidal ideation'. Leave blank for 'risk not present.'"
)

output_format_prompt = (
    "Output should be STRICT JSON, containing: dictionary containing the type of risk with their output and explanation, formatted like this: {'risk_type': 'suicidality', 'risk_output': str, 'explanation': str}"
)

# Define output schema for a more general case
output_schema = {
    'key_to_extract': None,  # Set to None for direct output
    'value_mapping': None,   # Set to None for direct mapping
    'regex_pattern': r'^(0|1)$'  # Match the entire output for binary classification
}

# Set number of optimization iterations
iterations = 3

# Define model providers and models for evaluation and optimization
eval_provider = "ollama"
eval_model = "llama3.1"
optim_provider = "ollama"
optim_model = "llama3.1"

# Path to the CSV file containing review data for evaluation
eval_datapath = "reviews.csv"

# Load and prepare data
eval_data = pd.read_csv(eval_datapath, encoding='ISO-8859-1', usecols=['Text', 'Sentiment'])
eval_data.columns = ['text', 'label']
# Randomly select 5 positive and 5 negative samples
eval_data = (
    eval_data.groupby('label')
    .apply(lambda x: x.sample(n=5, random_state=42))
    .reset_index(drop=True)
)
# Shuffle the DataFrame randomly
eval_data = eval_data.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Evaluation data shape: {eval_data.shape}")
print(eval_data.head())

# Run the prompt optimization process
best_prompt, best_metrics = optimize_prompt(
    initial_prompt,
    output_format_prompt,
    eval_data,
    iterations,
    eval_provider=eval_provider,
    eval_model=eval_model,
    optim_provider=optim_provider,
    optim_model=optim_model,
    output_schema=output_schema
)

print("\nBest Prompt:")
print(best_prompt)
print("\nBest Metrics:")
print(best_metrics)

# After running the optimization process, you can analyze the results by checking 
# the generated log files in the `runs/prompt_optimization_logs_YYYYMMDD_HHMMSS` directory.