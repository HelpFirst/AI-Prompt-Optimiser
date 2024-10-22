import pandas as pd 
from .evaluation import evaluate_prompt
from .prompt_generation import generate_new_prompt, validate_and_improve_prompt
from .utils import (estimate_token_usage, estimate_cost, display_best_prompt,
                    display_comparison_table, log_final_results, select_model,
                    create_log_directory, log_initial_setup, display_metrics,
                    create_metric_entry, update_best_metrics)
from . import config
from rich import print as rprint
from rich.panel import Panel
from .dashboard_generator import generate_iteration_dashboard, generate_combined_dashboard
from .model_interface import get_model_output, get_analysis

def optimize_prompt(initial_prompt: str, output_format_prompt: str, eval_data: pd.DataFrame, iterations: int,
                    eval_provider: str = None, eval_model: str = None, eval_temperature: float = 0.7,
                    optim_provider: str = None, optim_model: str = None, optim_temperature: float = 0.9,
                    output_schema: dict = None, use_cache: bool = True,
                    fp_comments: str = "", fn_comments: str = "", tp_comments: str = "",
                    invalid_comments: str = "", validation_comments: str = "",
                    prompt_engineering_comments: str = "", experiment_name: str = None,
                    skip_prompt_validation: bool = False) -> tuple:
    """
    Optimize a prompt through iterative refinement and evaluation.

    This function performs the following steps:
    1. Select and configure the language model
    2. Estimate token usage and cost
    3. Iteratively evaluate and improve the prompt
    4. Log results and display the best prompt

    Args:
        initial_prompt (str): The starting prompt to be optimized
        output_format_prompt (str): Instructions for the desired output format, aimed to be enforced in each of the iterations
        eval_data (pd.DataFrame): Dataset used for evaluating prompt performance
        iterations (int): Number of optimization iterations to perform
        eval_provider (str, optional): Provider for the evaluation model
        eval_model (str, optional): Name of the evaluation model
        eval_temperature (float, optional): Temperature for the evaluation model. Defaults to 0.7.
        optim_provider (str, optional): Provider for the optimization model
        optim_model (str, optional): Name of the optimization model
        optim_temperature (float, optional): Temperature for the optimization model. Defaults to 0.9.
        output_schema (dict, optional): Schema for transforming and comparing output with the true label
        use_cache (bool, optional): Whether to use cached model outputs. Defaults to True.
        fp_comments (str, optional): Comments for false positives analysis. Defaults to "".
        fn_comments (str, optional): Comments for false negatives analysis. Defaults to "".
        tp_comments (str, optional): Comments for true positives analysis. Defaults to "".
        invalid_comments (str, optional): Comments for invalid outputs analysis. Defaults to "".
        validation_comments (str, optional): Comments for validation and improvement step. Defaults to "".
        prompt_engineering_comments (str, optional): Comments for prompt engineering. Defaults to "".
        experiment_name (str, optional): Name of the experiment to be included in the run folder name
        skip_prompt_validation (bool, optional): Whether to skip validation and improvement steps. Defaults to False.

    Returns:
        tuple: The best performing prompt and its associated metrics
    """
    # Select models for evaluation and optimization if not provided
    if not eval_provider or not eval_model:
        eval_provider, eval_model = select_model("evaluation")
    if not optim_provider or not optim_model:
        optim_provider, optim_model = select_model("optimization")
    
    config.set_models(eval_provider, eval_model, eval_temperature, optim_provider, optim_model, optim_temperature)

    # Get the selected models from config
    eval_provider, eval_model, eval_temperature = config.get_eval_model()
    optim_provider, optim_model, optim_temperature = config.get_optim_model()

    print(f"Selected evaluation provider: {eval_provider}")
    print(f"Selected evaluation model: {eval_model}")
    print(f"Evaluation temperature: {eval_temperature}")
    print(f"Selected optimization provider: {optim_provider}")
    print(f"Selected optimization model: {optim_model}")
    print(f"Optimization temperature: {optim_temperature}")

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
    log_dir = create_log_directory(experiment_name)

    # Log initial setup
    log_initial_setup(log_dir, initial_prompt, output_format_prompt, iterations, eval_data,
                      eval_provider, eval_model, eval_temperature,
                      optim_provider, optim_model, optim_temperature,
                      output_schema, use_cache,
                      fp_comments, fn_comments, tp_comments,
                      invalid_comments, validation_comments)
    

    # Main optimization loop
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Print the full prompt
        rprint(Panel(current_prompt, title="Current Full Prompt", expand=False, border_style="blue"))
        
        # Evaluate the current prompt
        results = evaluate_prompt(current_prompt, eval_data, output_schema, log_dir=log_dir, iteration=i+1, use_cache=use_cache,
                                  provider=eval_provider, model=eval_model, temperature=eval_temperature)
        
        # Initialize prompts_used in results
        results['prompts_used'] = {}
        
        # Display and log the results
        display_metrics(results, i+1)
        all_metrics.append(create_metric_entry(i+1, results))
        
        # Update the best prompt if necessary
        best_metrics, best_prompt = update_best_metrics(best_metrics, best_prompt, results, current_prompt)
        
        # Generate a new prompt for the next iteration, except for the last one
        if i < iterations - 1:
            previous_metrics = {
                'precision': results['precision'],
                'recall': results['recall'],
                'accuracy': results['accuracy'],
                'f1': results['f1'],
                'valid_predictions': results['valid_predictions'],
                'invalid_predictions': results['invalid_predictions'],
                'total_predictions': results['valid_predictions'] + results['invalid_predictions']
            }
            new_prompt, analyses, prompts_used = generate_new_prompt(
                current_prompt,
                output_format_prompt,
                results['false_positives'],
                results['false_negatives'],
                results['true_positives'],
                results['invalid_outputs'],
                previous_metrics,
                log_dir=log_dir,
                iteration=i+1,
                provider=optim_provider,
                model=optim_model,
                temperature=optim_temperature,
                fp_comments=fp_comments,
                fn_comments=fn_comments,
                tp_comments=tp_comments,
                invalid_comments=invalid_comments,
                prompt_engineering_comments=prompt_engineering_comments
            )
            
            # Add analyses and prompts used to results
            results.update(analyses)
            results['prompts_used'] = prompts_used
            
            # Validate and improve the new prompt if not skipped
            if not skip_prompt_validation:
                improved_prompt, validation_result = validate_and_improve_prompt(new_prompt, output_format_prompt, 
                                                                                 provider=optim_provider, model=optim_model, 
                                                                                 temperature=optim_temperature,
                                                                                 validation_comments=validation_comments)
                
                # Add validation result to results
                results['validation_result'] = validation_result
                results['new_prompt'] = new_prompt
                results['improved_prompt'] = improved_prompt

                current_prompt = improved_prompt
            else:
                # Skip validation and use the new prompt directly
                current_prompt = new_prompt
                results['validation_result'] = "Skipped"
                results['new_prompt'] = new_prompt
                results['improved_prompt'] = new_prompt

        # Generate and save iteration dashboard
        generate_iteration_dashboard(log_dir, i+1, results, current_prompt, output_format_prompt, initial_prompt)

    # Generate and save combined dashboard
    generate_combined_dashboard(log_dir, all_metrics, best_prompt, output_format_prompt)

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
