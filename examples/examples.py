import pandas as pd
from src.iterative_prompt_optimization.optimize import optimize_prompt

# Load your dataset
eval_data = pd.read_csv('examples/reviews.csv')

# Define your initial prompt
initial_prompt = """
You are an AI assistant trained to analyze product reviews and determine if they indicate a safety risk. 
A safety risk is any mention of a product malfunction, physical injury, or potential danger to the user.
Analyze the given product review and determine if it indicates a safety risk.
"""

# Define the output format instructions
output_format_prompt = """
Provide your analysis in the following JSON format:
{
    "risk_output": "risk present" or "risk not present",
    "explanation": "A brief explanation of your decision"
}
"""

# Define the output schema
output_schema = {
    'key_to_extract': 'risk_output',
    'value_mapping': {
        'risk present': 1,
        'risk not present': 0
    },
    'regex_pattern': r'"risk_output":\s*"(.*?)"',
    'use_json_mode': True
}

# Set the number of iterations
iterations = 5

# Define model providers and models for evaluation and optimization
eval_provider = "azure_openai"
eval_model = "gpt-35-turbo"
optim_provider = "azure_openai"
optim_model = "gpt-4"

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

# Print the results
print("\nBest Prompt:")
print(best_prompt)
print("\nBest Metrics:")
for key, value in best_metrics.items():
    if key not in ['predictions', 'false_positives', 'false_negatives']:
        print(f"{key}: {value}")