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
    Generates a new prompt for multiclass classification by analyzing correct and incorrect predictions.
    """
    print("\nAnalyzing predictions for multiclass classification...")
    
    total_predictions = previous_metrics['total_predictions']
    analyses = {}
    prompts_used = {}

    # Process all predictions to separate correct and incorrect ones
    correct_predictions = []
    incorrect_predictions = []
    
    # Use the results dictionary that contains all necessary information
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

    # Analyze Correct Predictions
    num_correct = len(correct_predictions)
    if num_correct > 0:
        correct_texts_and_cot = "\n\n".join(
            f"** Text {i+1}:\n{item['text']}\n\nPredicted Class: {item['predicted_class']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n"
            for i, item in enumerate(correct_predictions)
        )
        
        class_counts = {}
        for item in correct_predictions:
            class_counts[item['predicted_class']] = class_counts.get(item['predicted_class'], 0) + 1
        class_distribution = ", ".join(f"{k}: {v}" for k, v in class_counts.items())
        
        correct_percentage = (num_correct / total_predictions) * 100
        
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

    # Analyze Incorrect Predictions
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

    # Format per-class metrics
    per_class_metrics = format_per_class_metrics(previous_metrics)

    # Generate improved prompt
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
    
    new_prompt = get_analysis(provider, model, temperature, prompt_engineer_input)
    prompts_used['prompt_engineer_input'] = prompt_engineer_input
    
    if log_dir and iteration:
        log_prompt_generation_multiclass(  # Use the imported function
            log_dir,
            iteration,
            initial_prompt,
            correct_analysis=correct_analysis,
            incorrect_analysis=incorrect_analysis,
            new_prompt=new_prompt
        )

    return new_prompt, analyses, prompts_used

def format_per_class_metrics(metrics: dict) -> str:
    """Helper function to format per-class metrics."""
    if 'per_class_metrics' not in metrics:
        return "No per-class metrics available"
    
    return "\n".join(
        f"  {class_name}:\n    Precision: {metrics['precision']}\n    Recall: {metrics['recall']}\n    F1: {metrics['f1']}"
        for class_name, metrics in metrics['per_class_metrics'].items()
    )
