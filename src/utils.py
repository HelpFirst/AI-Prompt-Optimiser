import os
import re
import json
from datetime import datetime
import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from .config import PRICING, MODEL_OPTIONS, SELECTED_PROVIDER, MODEL_NAME
from pathlib import Path
import ast
import pandas as pd
from .dashboard_generator import (
    generate_iteration_dashboard,
    generate_combined_dashboard
)

console = Console()

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
        if not isinstance(text, str):  # Check if text is a string
            print(f"Warning: Expected string but got {type(text)}. Skipping this entry.")
            return 0  # Return 0 tokens for non-string entries
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

def display_prompt(prompt: str, title: str = "Prompt"):
    """Display a prompt in a panel with a title."""
    panel = Panel(prompt, title=title, expand=False)
    console.print(panel)

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
        "valid_predictions": metrics['valid_predictions'],
        "invalid_predictions": metrics['invalid_predictions']
    }
    
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def display_analysis(analysis: str, title: str = "Analysis"):
    """Display the analysis in a formatted panel."""
    analysis_panel = Panel(
        analysis,
        title=title,
        expand=False,
        border_style="bold",
        padding=(1, 1)
    )
    Console().print(analysis_panel)

def log_prompt_generation(log_dir: str, iteration: int, initial_prompt: str, 
                         fp_analysis: str, fn_analysis: str, tp_analysis: str, 
                         invalid_analysis: str, new_prompt: str,
                         prompts_used: dict = None) -> None:
    """
    Log the prompt generation process for a specific iteration.
    
    Args:
        log_dir (str): Directory for storing logs
        iteration (int): Current iteration number
        initial_prompt (str): The initial prompt
        fp_analysis (str): Analysis of false positives
        fn_analysis (str): Analysis of false negatives
        tp_analysis (str): Analysis of true positives
        invalid_analysis (str): Analysis of invalid outputs
        new_prompt (str): The newly generated prompt
        prompts_used (dict): Dictionary containing all prompts used in the analysis
    """
    log_file_path = os.path.join(log_dir, f"iteration_{iteration}_prompt_generation.json")
    log_data = {
        "initial_prompt": initial_prompt,
        "analysis_results": {
            "false_positives": fp_analysis,
            "false_negatives": fn_analysis,
            "true_positives": tp_analysis,
            "invalid_outputs": invalid_analysis
        },
        "prompts_for_analysis": {
            "false_positives": prompts_used.get('fp_prompt', 'No prompt used'),
            "false_negatives": prompts_used.get('fn_prompt', 'No prompt used'),
            "true_positives": prompts_used.get('tp_prompt', 'No prompt used'),
            "invalid_outputs": prompts_used.get('invalid_prompt', 'No prompt used'),
            "prompt_engineering": prompts_used.get('prompt_engineer_input', 'No prompt used')
        } if prompts_used else {},
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
    comparison_table.add_column("Valid Predictions", justify="right")
    comparison_table.add_column("Invalid Predictions", justify="right")

    max_values = get_max_values(all_metrics)

    for metrics in all_metrics:
        comparison_table.add_row(
            str(metrics['iteration']),
            format_metric(metrics['precision'], max_values['precision']),
            format_metric(metrics['recall'], max_values['recall']),
            format_metric(metrics['accuracy'], max_values['accuracy']),
            format_metric(metrics['f1'], max_values['f1']),
            format_metric(metrics['valid_predictions'], max_values['valid_predictions']),
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
        'valid_predictions': max(m['valid_predictions'] for m in all_metrics),
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

def create_log_directory(experiment_name: str = None):
    """
    Create a directory for logging with a human-readable timestamp within a 'runs' folder.
    
    This function:
    1. Creates a 'runs' folder in the current working directory if it doesn't exist
    2. Creates a timestamped folder within 'runs' for the current optimization run
    
    Args:
        experiment_name (str, optional): Name of the experiment to be included in the run folder name

    Returns:
        str: Path to the newly created log directory
    """
    # Get the current working directory
    current_dir = Path.cwd()
    
    # Create the 'runs' folder if it doesn't exist
    runs_folder = current_dir / "runs"
    runs_folder.mkdir(exist_ok=True)
    
    # Create a human-readable timestamp
    timestamp = datetime.now().strftime("%a_%d-%b-%Y_%H-%M-%S")
    
    # Create the run folder name
    if experiment_name:
        run_folder_name = f"{experiment_name}_{timestamp}"
    else:
        run_folder_name = f"prompt_optimization_{timestamp}"
    
    log_dir = runs_folder / run_folder_name
    log_dir.mkdir()
    
    return str(log_dir)

def log_initial_setup(log_dir: str, initial_prompt: str, output_format_prompt: str, iterations: int, eval_data,
                      eval_provider: str, eval_model: str, eval_temperature: float,
                      optim_provider: str, optim_model: str, optim_temperature: float,
                      output_schema: dict = None, use_cache: bool = True,
                      fp_comments: str = "", fn_comments: str = "", tp_comments: str = "",
                      invalid_comments: str = "", validation_comments: str = ""):
    """Log the initial setup of the optimization process."""
    with open(os.path.join(log_dir, "initial_setup.json"), 'w') as f:
        json.dump({
            "initial_prompt": initial_prompt,
            "output_format_prompt": output_format_prompt,
            "iterations": iterations,
            "eval_data_shape": eval_data.shape,
            "evaluation_model": {
                "provider": eval_provider,
                "model": eval_model,
                "temperature": eval_temperature
            },
            "optimization_model": {
                "provider": optim_provider,
                "model": optim_model,
                "temperature": optim_temperature
            },
            "output_schema": output_schema,
            "use_cache": use_cache,
            "comments": {
                "false_positives": fp_comments,
                "false_negatives": fn_comments,
                "true_positives": tp_comments,
                "invalid_outputs": invalid_comments,
                "validation": validation_comments
            }
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

def display_confusion_matrix(conf_matrix: list):
    """
    Display a nicely formatted confusion matrix in the console with horizontal lines
    and bold diagonal elements.
    
    Args:
        conf_matrix (list): 2D list representing the confusion matrix
    """
    if not conf_matrix:
        return

    console = Console()
    table = Table(title="Confusion Matrix", show_header=True, show_lines=True)
    
    # Add columns
    table.add_column("True↓/Pred→", no_wrap=True)
    for i in range(len(conf_matrix)):
        table.add_column(str(i), justify="right")
        
    # Add rows
    for i in range(len(conf_matrix)):
        row = [str(i)]  # First column is the true label
        for j in range(len(conf_matrix)):
            # Bold the diagonal elements
            if i == j:
                row.append(f"[bold]{conf_matrix[i][j]}[/bold]")
            else:
                row.append(str(conf_matrix[i][j]))
        table.add_row(*row)
    
    console.print(table)
    print()

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
    metrics_table.add_row("", "", style="dim", end_section=True)  # Horizontal line separator
    metrics_table.add_row("Valid Predictions", str(results['valid_predictions']))
    metrics_table.add_row("Invalid Predictions", str(results['invalid_predictions']))
    
    console.print(metrics_table)
    
    # Display confusion matrix if available
    if results.get('confusion_matrix'):
        display_confusion_matrix(results['confusion_matrix'])

def create_metric_entry(iteration: int, results: dict):
    """Create a metric entry for a specific iteration."""
    return {
        'iteration': iteration,
        'precision': results['precision'],
        'recall': results['recall'],
        'accuracy': results['accuracy'],
        'f1': results['f1'],
        'valid_predictions': results['valid_predictions'],
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

    This function attempts to parse the raw output using various methods, extract the relevant
    information, and compare it with the provided label. It handles different output formats
    and provides detailed error messages for debugging.

    Args:
        raw_output (str): The raw output from the model
        label (int): The true label from eval_data
        output_schema (dict): A schema defining how to transform the output

    Returns:
        tuple: (transformed_output, is_correct, is_valid, chain_of_thought)
    """
    # Extract necessary information from the output schema
    # These values will be used throughout the function for parsing and transformation
    key_to_extract = output_schema.get('key_to_extract')
    value_mapping = output_schema.get('value_mapping')
    regex_pattern = output_schema.get('regex_pattern')
    chain_of_thought_key = output_schema.get('chain_of_thought_key')
    chain_of_thought_regex = output_schema.get('chain_of_thought_regex')

    # Define helper functions for each parsing method
    def try_ast_literal_eval():
        """
        Attempt to parse the raw output using ast.literal_eval()
        This method is safe for evaluating strings containing Python expressions.
        """
        try:
            return ast.literal_eval(raw_output), None
        except (ValueError, SyntaxError):
            print("Python literal evaluation failed...")
            return None, None

    def try_json_loads():
        """
        Attempt to parse the raw output using json.loads()
        This method is used if the output is in JSON format.
        """
        try:
            return json.loads(raw_output), None
        except json.JSONDecodeError:
            print("JSON parsing failed...")
            return None, None

    def try_json_like_structure():
        """
        Attempt to extract and parse a JSON-like structure from the raw output
        This method is used if the output contains a JSON-like structure within other text.
        """
        json_match = re.search(r'\{[^}]+\}', raw_output)
        if json_match:
            try:
                return json.loads(json_match.group()), None
            except json.JSONDecodeError:
                print("Failed to parse JSON-like structure")
        return None, None

    def try_regex_extraction():
        """
        Attempt to extract values using regex patterns
        This method is used as a last resort if other parsing methods fail.
        """
        value_match = re.search(regex_pattern, raw_output)
        cot_match = re.search(chain_of_thought_regex, raw_output) if chain_of_thought_regex else None
        if value_match:
            extracted_value = value_match.group(1)
            chain_of_thought = cot_match.group(1) if cot_match else "N/A"
            print(f"Regex extraction successful. Extracted value: '{extracted_value}'")
            return extracted_value, chain_of_thought
        return None, "N/A"

    # Try each parsing method in order
    # This allows for flexibility in handling different output formats
    for parse_method in [try_ast_literal_eval, try_json_loads, try_json_like_structure, try_regex_extraction]:
        parsed_output, chain_of_thought = parse_method()
        if parsed_output is not None:
            # print(f"Parsing successful using {parse_method.__name__}")
            break  # Exit the loop if parsing is successful
    else:
        # This block will only execute if the loop completes without a break
        print("All parsing methods failed!")
        return None, False, False, "N/A"

    # If we've reached this point, parsing was successful
    # print(f"Final parsed output: {parsed_output}")

    # Extract the relevant value and chain of thought
    # Handle both dictionary and non-dictionary parsed outputs
    if isinstance(parsed_output, dict):
        extracted_value = parsed_output.get(key_to_extract)
        chain_of_thought = parsed_output.get(chain_of_thought_key, chain_of_thought or "N/A")
    else:
        extracted_value = parsed_output
        # chain_of_thought is already set by try_regex_extraction if applicable

    # Handle cases where extraction failed
    if extracted_value is None:
        print(f"Extracted value is None. Raw output: {raw_output}")
        return None, False, False, chain_of_thought

    # Transform the extracted value
    # This step maps the extracted value to the desired output format
    if value_mapping:
        if isinstance(extracted_value, str):
            # Check the value directly first
            transformed_output = value_mapping.get(extracted_value)
            if transformed_output is None:
                # If not found, try normalized version
                normalized_value = extracted_value.lower().replace(" ", "_")
                transformed_output = value_mapping.get(normalized_value)
        elif isinstance(extracted_value, int):
            # If the extracted value is already an integer, use it directly
            transformed_output = extracted_value
        else:
            print(f"Unexpected value type: {type(extracted_value)}. Value: {extracted_value}")
            return None, False, False, chain_of_thought
    else:
        # If no value mapping is provided, attempt to convert the extracted value to an integer
        if isinstance(extracted_value, int):
            transformed_output = extracted_value
        else:
            try:
                transformed_output = int(extracted_value)
            except ValueError:
                print(f"Invalid output: Unable to convert '{extracted_value}' to int")
                return None, False, False, chain_of_thought

    # Final validity check and comparison with the label
    if transformed_output is not None:
        is_correct = (transformed_output == label)
        return transformed_output, is_correct, True, chain_of_thought
    else:
        print(f"Invalid output: Extracted value '{extracted_value}' not found in value_mapping")
        return None, False, False, chain_of_thought

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
            "valid_predictions": results['valid_predictions'],
            "invalid_predictions": results['invalid_predictions']
        },
        "false_positives": [fp['text'] for fp in results['false_positives']],
        "false_negatives": [fn['text'] for fn in results['false_negatives']],
        "eval_data_shape": eval_data.shape
    }
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def detect_problem_type(eval_data: pd.DataFrame, output_schema: dict) -> str:
    """
    Detect whether the classification problem is binary or multiclass.
    
    Args:
        eval_data (pd.DataFrame): The evaluation dataset containing labels
        output_schema (dict): Schema defining the output format and value mappings
        
    Returns:
        str: 'binary' or 'multiclass'
    """
    # Check unique labels in the evaluation data
    unique_labels = eval_data['label'].nunique()
    
    # Check value mapping length in the output schema
    value_mapping = output_schema.get('value_mapping', {})
    value_mapping_length = len(value_mapping)
    
    # If either the data or schema indicates more than 2 classes, it's multiclass
    if unique_labels > 2 or value_mapping_length > 2:
        return 'multiclass'
    else:
        return 'binary'

# Add this new function in utils.py
def log_prompt_generation_multiclass(log_dir: str, iteration: int, initial_prompt: str, 
                                   correct_analysis: str = None, incorrect_analysis: str = None, 
                                   new_prompt: str = None, prompts_used: dict = None) -> None:
    """
    Log the prompt generation process for multiclass classification.
    
    Args:
        log_dir (str): Directory for storing logs
        iteration (int): Current iteration number
        initial_prompt (str): The initial prompt
        correct_analysis (str): Analysis of correct predictions
        incorrect_analysis (str): Analysis of incorrect predictions
        new_prompt (str): The newly generated prompt
        prompts_used (dict): Dictionary containing all prompts used in the analysis
    """
    log_file_path = os.path.join(log_dir, f"iteration_{iteration}_prompt_generation.json")
    log_data = {
        "initial_prompt": initial_prompt,
        "correct_analysis": correct_analysis,
        "incorrect_analysis": incorrect_analysis,
        "new_prompt": new_prompt,
        "prompts_for_analysis": {
            "correct_prompt": prompts_used.get('correct_prompt', 'No prompt used'),
            "incorrect_prompt": prompts_used.get('incorrect_prompt', 'No prompt used'),
            "prompt_engineer_input": prompts_used.get('prompt_engineer_input', 'No prompt used')
        } if prompts_used else {}
    }
    with open(log_file_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def regenerate_dashboards(path_to_data: str, is_binary: bool = True):
    """
    Regenerate dashboards for a specific experiment.
    
    Args:
        path_to_data: Path to the experiment data directory
        is_binary: Whether this is a binary classification problem
    """
    # Load experiment data
    with open(os.path.join(path_to_data, 'experiment_config.json'), 'r') as f:
        config = json.load(f)
    
    # Get number of iterations from directory contents
    iteration_files = [f for f in os.listdir(path_to_data) 
                      if f.startswith('iteration_') and f.endswith('_evaluation.json')]
    num_iterations = len(iteration_files)
    
    # Load metrics for all iterations
    all_metrics = []
    for i in range(1, num_iterations + 1):
        with open(os.path.join(path_to_data, f'iteration_{i}_evaluation.json'), 'r') as f:
            metrics = json.load(f)
            metrics['iteration'] = i
            all_metrics.append(metrics)
    
    # Regenerate individual dashboards
    for i in range(1, num_iterations + 1):
        # Load iteration data
        with open(os.path.join(path_to_data, f'iteration_{i}_evaluation.json'), 'r') as f:
            results = json.load(f)
        
        with open(os.path.join(path_to_data, f'iteration_{i}_prompt_generation.json'), 'r') as f:
            prompt_data = json.load(f)
        
        # Generate dashboard
        generate_iteration_dashboard(
            log_dir=path_to_data,
            iteration=i,
            results=results,
            current_prompt=prompt_data['initial_prompt'],
            output_format_prompt=config['output_format_prompt'],
            initial_prompt=config['initial_prompt'],
            is_binary=is_binary
        )
    
    # Generate combined dashboard
    generate_combined_dashboard(
        log_dir=path_to_data,
        all_metrics=all_metrics,
        best_prompt=prompt_data['new_prompt'],  # Use the last generated prompt
        output_format_prompt=config['output_format_prompt'],
        is_binary=is_binary
    )

def get_result_symbol(is_correct: bool, is_valid: bool, true_label: int, pred_label: int, problem_type: str = "binary") -> str:
    """
    Generate a result symbol for display in the dashboard.
    
    Args:
        is_correct (bool): Whether the prediction was correct
        is_valid (bool): Whether the output format was valid
        true_label (int): The true label
        pred_label (int): The predicted label
        problem_type (str): Type of classification ("binary" or "multiclass")
        
    Returns:
        str: A formatted string with emoji and result type 
             Binary: "✅ (TP)", "✅ (TN)", "❌ (FP)", "❌ (FN)"
             Multiclass: "✅ (Correct)", "❌ (Incorrect)"
    """
    if not is_valid:
        return "❌ (Invalid)"
        
    if problem_type == "binary":
        if is_correct:
            return "✅ (TP)" if true_label == 1 else "✅ (TN)"
        else:
            return "❌ (FP)" if pred_label == 1 else "❌ (FN)"
    else:  # multiclass
        return "✅ (Correct)" if is_correct else f"❌ (Predicted {pred_label}, True {true_label})"
