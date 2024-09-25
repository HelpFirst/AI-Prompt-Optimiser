import pandas as pd 
from .evaluation import evaluate_prompt
from .prompt_generation import generate_new_prompt
from .utils import (estimate_token_usage, estimate_cost, display_best_prompt,
                    display_comparison_table, log_final_results, select_model,
                    create_log_directory, log_initial_setup, display_metrics,
                    create_metric_entry, update_best_metrics)
from . import config

def optimize_prompt(initial_prompt: str, output_format_prompt: str, eval_data: pd.DataFrame, iterations: int,
                    eval_provider: str = None, eval_model: str = None,
                    optim_provider: str = None, optim_model: str = None,
                    output_schema: dict = None) -> tuple:
    """
    Optimize a prompt through iterative refinement and evaluation.

    This function performs the following steps:
    1. Select and configure the language model
    2. Estimate token usage and cost
    3. Iteratively evaluate and improve the prompt
    4. Log results and display the best prompt

    Args:
        initial_prompt (str): The starting prompt to be optimized
        output_format_prompt (str): Instructions for the desired output format
        eval_data (pd.DataFrame): Dataset used for evaluating prompt performance
        iterations (int): Number of optimization iterations to perform
        eval_provider (str, optional): Provider for the evaluation model
        eval_model (str, optional): Name of the evaluation model
        optim_provider (str, optional): Provider for the optimization model
        optim_model (str, optional): Name of the optimization model
        output_schema (dict, optional): Schema for transforming and comparing output

    Returns:
        tuple: The best performing prompt and its associated metrics
    """
    # Select models for evaluation and optimization if not provided
    if not eval_provider or not eval_model:
        eval_provider, eval_model = select_model("evaluation")
    if not optim_provider or not optim_model:
        optim_provider, optim_model = select_model("optimization")
    
    config.set_models(eval_provider, eval_model, optim_provider, optim_model)

    # Get the selected models from config
    eval_provider, eval_model = config.get_eval_model()
    optim_provider, optim_model = config.get_optim_model()

    print(f"Selected evaluation provider: {eval_provider}")
    print(f"Selected evaluation model: {eval_model}")
    print(f"Selected optimization provider: {optim_provider}")
    print(f"Selected optimization model: {optim_model}")

    # Estimate token usage and cost
    total_tokens = estimate_token_usage(initial_prompt, output_format_prompt, eval_data, iterations)
    estimated_cost = estimate_cost(total_tokens, eval_provider, eval_model)
    
    print(f"Estimated token usage: {total_tokens}")
    print(f"Estimated cost: {estimated_cost}")
    
    # Confirm with the user before proceeding
    print("\nDo you want to proceed with the optimization? (Y/N): ", end="", flush=True)
    proceed = input().strip().lower()
    if proceed != 'y':
        print("Optimization cancelled.")
        return None, None

    best_metrics = None
    best_prompt = initial_prompt
    current_prompt = initial_prompt
    all_metrics = []

    # Create a directory for logging
    log_dir = create_log_directory()

    # Log initial setup
    log_initial_setup(log_dir, initial_prompt, output_format_prompt, iterations, eval_data)
    
    # Define a default output schema if none is provided
    if output_schema is None:
        output_schema = {
            'key_to_extract': 'risk_output',
            'value_mapping': {
                'risk present': 1,
                'risk not present': 0
            },
            'regex_pattern': r"'risk_output':\s*'(.*?)'"
        }

    # Main optimization loop
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Evaluate the current prompt
        results = evaluate_prompt(current_prompt, output_format_prompt, eval_data, output_schema, log_dir=log_dir, iteration=i+1)
        
        # Display and log the results
        display_metrics(results, i+1)
        all_metrics.append(create_metric_entry(i+1, results))
        
        # Update the best prompt if necessary
        best_metrics, best_prompt = update_best_metrics(best_metrics, best_prompt, results, current_prompt)
        
        # Generate a new prompt for the next iteration, except for the last one
        if i < iterations - 1:
            current_prompt = generate_new_prompt(
                current_prompt,
                output_format_prompt,
                results['false_positives'],
                results['false_negatives'],
                log_dir=log_dir,
                iteration=i+1
            )
    
    # Display and log final results
    display_best_prompt(best_prompt, output_format_prompt)
    display_comparison_table(all_metrics)
    log_final_results(log_dir, best_prompt, output_format_prompt, best_metrics, all_metrics)

    return best_prompt, best_metrics

def select_model(purpose: str) -> tuple:
    """
    Interactively select a model provider and specific model.

    Args:
        purpose (str): The purpose of model selection (e.g., "evaluation" or "optimization")

    Returns:
        tuple: Selected provider and model name
    """
    print(f"\nSelect a model provider for {purpose}:")
    providers = list(config.MODEL_OPTIONS.keys())
    for i, provider in enumerate(providers):
        print(f"{i+1}. {provider}")
    
    provider_choice = int(input("Enter the number of your choice: ")) - 1
    selected_provider = providers[provider_choice]
    
    print(f"\nSelect a model from {selected_provider} for {purpose}:")
    models = config.MODEL_OPTIONS[selected_provider]
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    model_choice = int(input("Enter the number of your choice: ")) - 1
    selected_model = models[model_choice]
    
    return selected_provider, selected_model