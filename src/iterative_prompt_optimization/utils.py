import os
import re
import json
from datetime import datetime
import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from .config import PRICING, MODEL_OPTIONS, SELECTED_PROVIDER, MODEL_NAME

def estimate_token_usage(initial_prompt: str, output_format_prompt: str, eval_data, iterations: int):
    """
    Estimate the total token usage for the prompt optimization process.

    This function:
    1. Initializes the appropriate tokenizer based on the selected model
    2. Estimates tokens for the initial prompt, output format, and evaluation data
    3. Estimates tokens for analysis and new prompt generation
    4. Calculates total token usage across all iterations

    Args:
        initial_prompt (str): The starting prompt
        output_format_prompt (str): Instructions for the desired output format
        eval_data: Dataset used for evaluation
        iterations (int): Number of optimization iterations

    Returns:
        int: Estimated total token usage
    """
    # Initialize tokenizer based on the selected model
    tokenizer = _get_tokenizer()

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    initial_prompt_tokens = count_tokens(initial_prompt)
    output_format_tokens = count_tokens(output_format_prompt)
    
    # Estimate tokens for evaluation
    eval_tokens = sum(count_tokens(text) for text in eval_data['text'])
    eval_tokens_per_iteration = (initial_prompt_tokens + output_format_tokens + eval_tokens) * len(eval_data)
    
    # Estimate tokens for analysis and new prompt generation
    analysis_tokens = 1000  # Estimated fixed number of tokens for analysis
    new_prompt_tokens = 1000  # Estimated fixed number of tokens for new prompt generation
    
    total_tokens = 0
    for i in range(iterations):
        total_tokens += eval_tokens_per_iteration
        total_tokens += analysis_tokens
        total_tokens += new_prompt_tokens
        
        # Assume the prompt grows in size each iteration
        initial_prompt_tokens = int(initial_prompt_tokens * 1.2)  # 20% growth per iteration
    
    return total_tokens

def _get_tokenizer():
    """Get the appropriate tokenizer based on the selected model."""
    if SELECTED_PROVIDER in ["openai", "azure_openai"]:
        return tiktoken.encoding_for_model(MODEL_NAME)
    elif SELECTED_PROVIDER in ["anthropic", "google"]:
        return tiktoken.encoding_for_model("cl100k_base")
    else:  # ollama
        return tiktoken.encoding_for_model("gpt-3.5-turbo")

def estimate_cost(total_tokens: int, provider: str, model: str):
    """
    Estimate the cost based on the total number of tokens and selected model.

    Args:
        total_tokens (int): Total number of tokens
        provider (str): The AI provider (e.g., "openai", "anthropic")
        model (str): The specific model being used

    Returns:
        str: Estimated cost as a string
    """
    if provider == "ollama":
        return "$0 API Costs - Running on Local Hardware"
    
    cost_per_1k_tokens = PRICING[provider][model]
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    return f"${estimated_cost:.2f}"

def display_prompt(full_prompt: str):
    """Display the full prompt in a formatted panel."""
    console = Console()
    prompt_panel = Panel(
        full_prompt,
        title="System Prompt",
        expand=False,
        border_style="yellow",
        padding=(1, 1)
    )
    console.print(prompt_panel)

def create_log_file_path(log_dir: str, iteration: int):
    """Create the log file path for a specific iteration."""
    return os.path.join(log_dir, f"iteration_{iteration}_evaluation.json")

def initialize_log_data(full_prompt: str):
    """Initialize the log data structure."""
    return {
        "prompt": full_prompt,
        "evaluations": []
    }

def log_results(log_file_path: str, log_data: dict, metrics: dict):
    """Log the evaluation results to a file."""
    log_data["metrics"] = {
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "accuracy": metrics['accuracy'],
        "f1": metrics['f1'],
        "invalid_predictions": metrics['invalid_predictions']
    }
    
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def display_analysis(analysis: str):
    """Display the analysis of misclassifications in a formatted panel."""
    analysis_panel = Panel(
        analysis,
        title="Analysis of Misclassifications",
        expand=False,
        border_style="bold",
        padding=(1, 1)
    )
    Console().print(analysis_panel)

def log_prompt_generation(log_dir: str, iteration: int, initial_prompt: str, analysis: str, new_prompt: str):
    """Log the prompt generation process for a specific iteration."""
    log_file_path = os.path.join(log_dir, f"iteration_{iteration}_prompt_generation.json")
    log_data = {
        "initial_prompt": initial_prompt,
        "analysis": analysis,
        "new_prompt": new_prompt
    }
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def display_best_prompt(best_prompt: str, output_format_prompt: str):
    """Display the best performing prompt."""
    console = Console()
    console.print(Panel.fit("[bold green]Best Prompt:[/bold green]"))
    console.print(best_prompt)
    console.print(Panel.fit("[bold green]Output Format:[/bold green]"))
    console.print(output_format_prompt)

def display_comparison_table(all_metrics: list):
    """Display a comparison table of metrics for all iterations."""
    console = Console()
    comparison_table = Table(title="Comparison of All Iterations", show_header=True)
    comparison_table.add_column("Iteration", justify="center")
    comparison_table.add_column("Precision", justify="right")
    comparison_table.add_column("Recall", justify="right")
    comparison_table.add_column("Accuracy", justify="right")
    comparison_table.add_column("F1-score", justify="right")
    comparison_table.add_column("Invalid Predictions", justify="right")

    max_values = get_max_values(all_metrics)

    for metrics in all_metrics:
        comparison_table.add_row(
            str(metrics['iteration']),
            format_metric(metrics['precision'], max_values['precision']),
            format_metric(metrics['recall'], max_values['recall']),
            format_metric(metrics['accuracy'], max_values['accuracy']),
            format_metric(metrics['f1'], max_values['f1']),
            format_metric(metrics['invalid_predictions'], max_values['invalid_predictions'], is_min=True)
        )

    console.print(comparison_table)

def get_max_values(all_metrics: list):
    """Get the maximum values for each metric across all iterations."""
    return {
        'precision': max(m['precision'] for m in all_metrics),
        'recall': max(m['recall'] for m in all_metrics),
        'accuracy': max(m['accuracy'] for m in all_metrics),
        'f1': max(m['f1'] for m in all_metrics),
        'invalid_predictions': min(m['invalid_predictions'] for m in all_metrics)
    }

def format_metric(value: float, max_value: float, is_min: bool = False):
    """Format a metric value, highlighting it if it's the maximum (or minimum for invalid predictions)."""
    if (not is_min and value == max_value) or (is_min and value == max_value):
        return f"[bold]{value:.4f}[/bold]"
    return f"{value:.4f}"

def log_final_results(log_dir: str, best_prompt: str, output_format_prompt: str, best_metrics: dict, all_metrics: list):
    """Log the final results of the optimization process."""
    with open(os.path.join(log_dir, "final_results.json"), 'w') as f:
        json.dump({
            "best_prompt": best_prompt + "\n\n" + output_format_prompt,
            "best_metrics": {k: v for k, v in best_metrics.items() if k not in ['false_positives', 'false_negatives', 'predictions']},
            "all_metrics": all_metrics
        }, f, indent=2)

    print(f"\nAll logs saved in directory: {log_dir}")

def create_log_directory():
    """
    Create a directory for logging with a timestamp within a 'runs' folder.
    
    This function:
    1. Creates a 'runs' folder in the project root if it doesn't exist
    2. Creates a timestamped folder within 'runs' for the current optimization run
    
    Returns:
        str: Path to the newly created log directory
    """
    # Get the project root directory (assuming utils.py is in src/iterative_prompt_optimization/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create the 'runs' folder if it doesn't exist
    runs_folder = os.path.join(project_root, "runs")
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)
    
    # Create a timestamped folder for the current run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(runs_folder, f"prompt_optimization_logs_{timestamp}")
    os.makedirs(log_dir)
    
    return log_dir

def log_initial_setup(log_dir: str, initial_prompt: str, output_format_prompt: str, iterations: int, eval_data):
    """Log the initial setup of the optimization process."""
    with open(os.path.join(log_dir, "initial_setup.json"), 'w') as f:
        json.dump({
            "initial_prompt": initial_prompt,
            "output_format_prompt": output_format_prompt,
            "iterations": iterations,
            "eval_data_shape": eval_data.shape
        }, f, indent=2)

def select_model():
    """Interactively select a model provider and specific model."""
    print("Select a model provider:")
    providers = list(MODEL_OPTIONS.keys())
    for i, provider in enumerate(providers):
        print(f"{i+1}. {provider}")
    
    provider_choice = int(input("Enter the number of your choice: ")) - 1
    selected_provider = providers[provider_choice]
    
    print(f"\nSelect a model from {selected_provider}:")
    models = MODEL_OPTIONS[selected_provider]
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    model_choice = int(input("Enter the number of your choice: ")) - 1
    selected_model = models[model_choice]
    
    return selected_provider, selected_model

def display_metrics(results: dict, iteration: int):
    """Display the evaluation metrics for a specific iteration."""
    console = Console()
    metrics_table = Table(title=f"Evaluation Metrics - Iteration {iteration}", show_header=True)
    metrics_table.add_column("Metric", no_wrap=True)
    metrics_table.add_column("Value", justify="right")
    
    metrics_table.add_row("Precision", f"{results['precision']:.4f}")
    metrics_table.add_row("Recall", f"{results['recall']:.4f}")
    metrics_table.add_row("Accuracy", f"{results['accuracy']:.4f}")
    metrics_table.add_row("F1-score", f"{results['f1']:.4f}")
    metrics_table.add_row("Invalid Predictions", str(results['invalid_predictions']))
    
    console.print(metrics_table)

def create_metric_entry(iteration: int, results: dict):
    """Create a metric entry for a specific iteration."""
    return {
        'iteration': iteration,
        'precision': results['precision'],
        'recall': results['recall'],
        'accuracy': results['accuracy'],
        'f1': results['f1'],
        'invalid_predictions': results['invalid_predictions']
    }

def update_best_metrics(best_metrics: dict, best_prompt: str, results: dict, current_prompt: str):
    """Update the best metrics and prompt if the current results are better."""
    if best_metrics is None or results['f1'] > best_metrics['f1']:
        return results, current_prompt
    return best_metrics, best_prompt

def transform_and_compare_output(raw_output, label, output_schema):
    """
    Transform the raw output according to the provided schema and compare with the label.

    Args:
        raw_output (str): The raw output from the model
        label (int): The true label from eval_data
        output_schema (dict): A schema defining how to transform the output

    Returns:
        tuple: (transformed_output, is_correct, is_valid)
    """
    key_to_extract = output_schema.get('key_to_extract')
    value_mapping = output_schema.get('value_mapping')
    regex_pattern = output_schema.get('regex_pattern')

    if key_to_extract:
        # Original case: extract from JSON or use regex for specific key
        json_match = re.search(r'\{[^}]+\}', raw_output)
        if json_match:
            try:
                parsed_output = json.loads(json_match.group())
                extracted_value = parsed_output.get(key_to_extract)
            except json.JSONDecodeError:
                print(f"JSON Decode Error: Unable to parse extracted JSON")
                return None, False, False
        else:
            # If JSON parsing fails, try to find the value directly
            value_match = re.search(regex_pattern, raw_output)
            if value_match:
                extracted_value = value_match.group(1)
            else:
                print(f"Unable to find {key_to_extract} in the raw output")
                return None, False, False
    else:
        # New case: direct binary classification
        value_match = re.search(regex_pattern, raw_output.strip())
        if value_match:
            extracted_value = value_match.group(1)
        else:
            print(f"Unable to match the output pattern: {regex_pattern}")
            return None, False, False

    if value_mapping:
        # Use value mapping if provided
        normalized_value = extracted_value.lower().replace(" ", "_")
        transformed_output = value_mapping.get(normalized_value)
    else:
        # Direct mapping for binary classification
        try:
            transformed_output = int(extracted_value)
        except ValueError:
            print(f"Invalid output: Unable to convert '{extracted_value}' to int")
            return None, False, False

    if transformed_output is not None:
        is_correct = (transformed_output == label)
        return transformed_output, is_correct, True
    else:
        print(f"Invalid output: Extracted value '{extracted_value}' not found in value_mapping")
        return None, False, False

def log_evaluation_results(log_dir: str, iteration: int, results: dict, eval_data):
    """Log the evaluation results for a specific iteration."""
    log_file_path = os.path.join(log_dir, f"iteration_{iteration}_evaluation.json")
    log_data = {
        "iteration": iteration,
        "metrics": {
            "precision": results['precision'],
            "recall": results['recall'],
            "accuracy": results['accuracy'],
            "f1": results['f1'],
            "invalid_predictions": results['invalid_predictions']
        },
        "false_positives": [fp['text'] for fp in results['false_positives']],
        "false_negatives": [fn['text'] for fn in results['false_negatives']],
        "eval_data_shape": eval_data.shape
    }
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)