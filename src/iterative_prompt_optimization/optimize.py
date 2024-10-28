# Main module for prompt optimization process
# Handles the iterative improvement of prompts through evaluation and analysis

import pandas as pd 
from .evaluation import evaluate_prompt
from .prompt_generation import generate_new_prompt, validate_and_improve_prompt
from .prompt_generation_multiclass import generate_new_prompt_multiclass
from .utils import (
    estimate_token_usage, estimate_cost, display_best_prompt,
    display_comparison_table, log_final_results, select_model,
    create_log_directory, log_initial_setup, display_metrics,
    create_metric_entry, update_best_metrics, detect_problem_type
)
from . import config
from rich import print as rprint
from rich.panel import Panel
from .dashboard_generator import generate_iteration_dashboard, generate_combined_dashboard
from .model_interface import get_model_output, get_analysis
from .dashboard_generator_multiclass import (
    generate_iteration_dashboard_multiclass,
    generate_combined_dashboard_multiclass
)

def optimize_prompt(initial_prompt: str, output_format_prompt: str, eval_data: pd.DataFrame, 
                   iterations: int, eval_provider: str = None, eval_model: str = None, 
                   eval_temperature: float = 0.7, optim_provider: str = None, 
                   optim_model: str = None, optim_temperature: float = 0.9,
                   output_schema: dict = None, use_cache: bool = True,
                   fp_comments: str = "", fn_comments: str = "", tp_comments: str = "",
                   invalid_comments: str = "", validation_comments: str = "",
                   prompt_engineering_comments: str = "", experiment_name: str = None,
                   skip_prompt_validation: bool = False) -> tuple:
    """
    Main function for optimizing prompts through iterative refinement.
    
    This function orchestrates the entire optimization process:
    1. Detects problem type (binary/multiclass)
    2. Sets up models and estimates costs
    3. Runs iterations of evaluation and improvement
    4. Generates dashboards and logs results
    
    Args:
        initial_prompt: Starting prompt to optimize
        output_format_prompt: Instructions for output formatting
        eval_data: Dataset for evaluation
        iterations: Number of optimization iterations
        eval_provider: Provider for evaluation model
        eval_model: Model for evaluation
        eval_temperature: Temperature for evaluation
        optim_provider: Provider for optimization model
        optim_model: Model for optimization
        optim_temperature: Temperature for optimization
        output_schema: Schema for output parsing
        use_cache: Whether to use response caching
        *_comments: Additional guidance for different analyses
        experiment_name: Name for the experiment run
        skip_prompt_validation: Whether to skip validation step
        
    Returns:
        tuple: (best_prompt, best_metrics)
    """
    # Detect whether this is binary or multiclass classification
    problem_type = detect_problem_type(eval_data, output_schema)
    print(f"\nDetected problem type: {problem_type}")
    
    # Select and configure models if not provided
    if not eval_provider or not eval_model:
        eval_provider, eval_model = select_model("evaluation")
    if not optim_provider or not optim_model:
        optim_provider, optim_model = select_model("optimization")
    
    # Set up global configuration
    config.set_models(eval_provider, eval_model, eval_temperature, 
                     optim_provider, optim_model, optim_temperature)

    # Display selected models and settings
    eval_provider, eval_model, eval_temperature = config.get_eval_model()
    optim_provider, optim_model, optim_temperature = config.get_optim_model()
    print(f"Selected evaluation provider: {eval_provider}")
    print(f"Selected evaluation model: {eval_model}")
    print(f"Evaluation temperature: {eval_temperature}")
    print(f"Selected optimization provider: {optim_provider}")
    print(f"Selected optimization model: {optim_model}")
    print(f"Optimization temperature: {optim_temperature}")

    # Estimate resource usage and cost
    total_tokens = estimate_token_usage(initial_prompt, output_format_prompt, eval_data, iterations)
    estimated_cost = estimate_cost(total_tokens, eval_provider, eval_model)
    print(f"Estimated token usage: {total_tokens}")
    print(f"Estimated cost: {estimated_cost}")
    
    # Get user confirmation before proceeding
    print("\nDo you want to proceed with the optimization? (Y/N): ", end="", flush=True)
    proceed = input().strip().lower()
    if proceed != 'y':
        print("Optimization cancelled.")
        return None, None

    # Initialize tracking variables
    best_metrics = None
    best_prompt = initial_prompt
    current_prompt = initial_prompt
    all_metrics = []

    # Set up logging directory and initial configuration
    log_dir = create_log_directory(experiment_name)
    log_initial_setup(log_dir, initial_prompt, output_format_prompt, iterations, eval_data,
                     eval_provider, eval_model, eval_temperature,
                     optim_provider, optim_model, optim_temperature,
                     output_schema, use_cache,
                     fp_comments, fn_comments, tp_comments,
                     invalid_comments, validation_comments)

    # Main optimization loop
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        rprint(Panel(current_prompt, title="Current Full Prompt", expand=False, border_style="blue"))
        
        # Evaluate current prompt
        results = evaluate_prompt(
            current_prompt, 
            eval_data, 
            output_schema, 
            problem_type=problem_type,  
            log_dir=log_dir, 
            iteration=i+1, 
            use_cache=use_cache,
            provider=eval_provider, 
            model=eval_model, 
            temperature=eval_temperature
        )
        
        # Track results and display metrics
        results['prompts_used'] = {}
        display_metrics(results, i+1)
        all_metrics.append(create_metric_entry(i+1, results))
        best_metrics, best_prompt = update_best_metrics(best_metrics, best_prompt, results, current_prompt)

        # Generate new prompt if not final iteration
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
            if problem_type == "binary":
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
            else:  # problem_type == "multiclass"
                new_prompt, analyses, prompts_used = generate_new_prompt_multiclass(
                    initial_prompt=current_prompt,
                    output_format_prompt=output_format_prompt,
                    results=results,
                    previous_metrics=previous_metrics,
                    log_dir=log_dir,
                    iteration=i+1,
                    provider=optim_provider,
                    model=optim_model,
                    temperature=optim_temperature,
                    correct_comments="",
                    incorrect_comments="",
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

        # Generate and save iteration dashboard based on problem type
        if problem_type == "binary":
            generate_iteration_dashboard(log_dir, i+1, results, current_prompt, 
                                      output_format_prompt, initial_prompt)
        else:  # multiclass
            generate_iteration_dashboard_multiclass(log_dir, i+1, results, current_prompt, 
                                                 output_format_prompt, initial_prompt)

    # Generate and save combined dashboard based on problem type
    if problem_type == "binary":
        generate_combined_dashboard(log_dir, all_metrics, best_prompt, output_format_prompt)
    else:  # multiclass
        generate_combined_dashboard_multiclass(log_dir, all_metrics, best_prompt, output_format_prompt)

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
