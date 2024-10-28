# Module for generating and improving prompts for multiclass classification tasks
# Handles analysis of correct/incorrect predictions and prompt optimization

from . import config
from .model_interface import get_analysis
from .utils import (
    display_analysis, 
    display_prompt,
    log_prompt_generation_multiclass
)
from .prompts_multiclass import (
    CORRECT_PREDICTIONS_ANALYSIS_PROMPT,
    INCORRECT_PREDICTIONS_ANALYSIS_PROMPT,
    PROMPT_ENGINEER_INPUT_MULTICLASS
)

def generate_new_prompt_multiclass(
    initial_prompt: str,
    output_format_prompt: str,
    results: dict,  # Pass the full results dictionary
    previous_metrics: dict,
    log_dir: str = None,
    iteration: int = None,
    provider: str = None,
    model: str = None,
    temperature: float = 0.9,
    correct_comments: str = "",
    incorrect_comments: str = "",
    prompt_engineering_comments: str = ""
) -> tuple:
    """
    Generates a new prompt for multiclass classification by analyzing prediction patterns.
    
    This function:
    1. Separates correct and incorrect predictions
    2. Analyzes patterns in both successful and failed classifications
    3. Generates a new improved prompt based on the analysis
    
    Args:
        initial_prompt: Starting prompt to be improved
        output_format_prompt: Instructions for output formatting
        results: Dictionary containing evaluation results and predictions
        previous_metrics: Metrics from the previous iteration
        log_dir: Directory for storing logs
        iteration: Current iteration number
        provider: AI provider for analysis
        model: Model name for analysis
        temperature: Temperature setting for generation
        correct_comments: Additional notes for correct predictions analysis
        incorrect_comments: Additional notes for incorrect predictions analysis
        prompt_engineering_comments: Additional guidance for prompt improvement

    Returns:
        tuple: (new_prompt, analyses_dict, prompts_used_dict)
    """
    print("\nAnalyzing predictions for multiclass classification...")
    
    total_predictions = previous_metrics['total_predictions']
    analyses = {}
    prompts_used = {}

    # Separate predictions into correct and incorrect groups for analysis
    correct_predictions = []
    incorrect_predictions = []
    
    # Process all predictions with their associated information
    for text, label, pred, cot in zip(
        results['texts'], 
        results['labels'], 
        results['predictions'], 
        results['chain_of_thought']
    ):
        if label == pred:
            correct_predictions.append({
                'text': text,
                'predicted_class': pred,
                'chain_of_thought': cot
            })
        else:
            incorrect_predictions.append({
                'text': text,
                'predicted_class': pred,
                'true_class': label,
                'chain_of_thought': cot
            })

    # Analyze correct predictions to identify successful patterns
    num_correct = len(correct_predictions)
    if num_correct > 0:
        # Format examples for analysis
        correct_texts_and_cot = "\n\n".join(
            f"** Text {i+1}:\n{item['text']}\n\nPredicted Class: {item['predicted_class']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n"
            for i, item in enumerate(correct_predictions)
        )
        
        # Calculate class distribution for successful predictions
        class_counts = {}
        for item in correct_predictions:
            class_counts[item['predicted_class']] = class_counts.get(item['predicted_class'], 0) + 1
        class_distribution = ", ".join(f"{k}: {v}" for k, v in class_counts.items())
        
        correct_percentage = (num_correct / total_predictions) * 100
        
        # Generate analysis of correct predictions
        correct_prompt = CORRECT_PREDICTIONS_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            correct_texts_and_cot=correct_texts_and_cot,
            total_predictions=total_predictions,
            num_correct=num_correct,
            correct_percentage=correct_percentage,
            class_distribution=class_distribution,
            correct_comments=correct_comments
        )
        
        correct_analysis = get_analysis(provider, model, temperature, correct_prompt)
    else:
        correct_analysis = "No correct predictions found in this iteration."
        correct_prompt = "No prompt used (no correct predictions)"

    # Analyze incorrect predictions to identify error patterns
    num_incorrect = len(incorrect_predictions)
    if num_incorrect > 0:
        incorrect_texts_and_cot = "\n\n".join(
            f"** Text {i+1}:\n{item['text']}\n\nPredicted Class: {item['predicted_class']}\nTrue Class: {item['true_class']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n"
            for i, item in enumerate(incorrect_predictions)
        )
        
        misclassification_counts = {}
        for item in incorrect_predictions:
            key = f"{item['true_class']}→{item['predicted_class']}"
            misclassification_counts[key] = misclassification_counts.get(key, 0) + 1
        misclassification_distribution = ", ".join(f"{k}: {v}" for k, v in misclassification_counts.items())
        
        incorrect_percentage = (num_incorrect / total_predictions) * 100
        
        incorrect_prompt = INCORRECT_PREDICTIONS_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            incorrect_texts_and_cot=incorrect_texts_and_cot,
            total_predictions=total_predictions,
            num_incorrect=num_incorrect,
            incorrect_percentage=incorrect_percentage,
            misclassification_distribution=misclassification_distribution,
            incorrect_comments=incorrect_comments
        )
        
        incorrect_analysis = get_analysis(provider, model, temperature, incorrect_prompt)
    else:
        incorrect_analysis = "No incorrect predictions found in this iteration."
        incorrect_prompt = "No prompt used (no incorrect predictions)"

    display_analysis(correct_analysis, "Correct Predictions Analysis")
    display_analysis(incorrect_analysis, "Incorrect Predictions Analysis")
    
    analyses['correct_analysis'] = correct_analysis
    analyses['incorrect_analysis'] = incorrect_analysis
    prompts_used['correct_prompt'] = correct_prompt
    prompts_used['incorrect_prompt'] = incorrect_prompt

    # Format metrics for prompt improvement
    per_class_metrics = format_per_class_metrics(previous_metrics)

    # Generate improved prompt using the analyses
    prompt_engineer_input = PROMPT_ENGINEER_INPUT_MULTICLASS.format(
        initial_prompt=initial_prompt,
        accuracy=previous_metrics['accuracy'],
        per_class_metrics=per_class_metrics,
        total_predictions=total_predictions,
        valid_predictions=previous_metrics['valid_predictions'],
        invalid_predictions=previous_metrics['invalid_predictions'],
        correct_analysis=correct_analysis,
        incorrect_analysis=incorrect_analysis,
        output_format_prompt=output_format_prompt,
        prompt_engineering_comments=prompt_engineering_comments
    )
    
    # Get the new improved prompt
    new_prompt = get_analysis(provider, model, temperature, prompt_engineer_input)
    prompts_used['prompt_engineer_input'] = prompt_engineer_input
    
    # Log the prompt generation process if logging is enabled
    if log_dir and iteration:
        log_prompt_generation_multiclass(
            log_dir,
            iteration,
            initial_prompt,
            correct_analysis=correct_analysis,
            incorrect_analysis=incorrect_analysis,
            new_prompt=new_prompt
        )

    return new_prompt, analyses, prompts_used

def format_per_class_metrics(metrics: dict) -> str:
    """
    Format per-class metrics into a readable string format.
    
    Args:
        metrics: Dictionary containing metrics for each class
        
    Returns:
        str: Formatted string of per-class metrics
    """
    if 'per_class_metrics' not in metrics:
        return "No per-class metrics available"
    
    return "\n".join(
        f"  {class_name}:\n    Precision: {metrics['precision']}\n    Recall: {metrics['recall']}\n    F1: {metrics['f1']}"
        for class_name, metrics in metrics['per_class_metrics'].items()
    )
