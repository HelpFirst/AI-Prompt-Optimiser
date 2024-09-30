from . import config
from .model_interface import get_analysis
from .utils import display_analysis, log_prompt_generation, display_prompt

def generate_new_prompt(initial_prompt: str, output_format_prompt: str, false_positives: list, false_negatives: list, true_positives: list, invalid_outputs: list, previous_metrics: dict, log_dir: str = None, iteration: int = None) -> str:
    """
    Generates a new prompt by incorporating false positives, false negatives, true positives, and invalid outputs analyses.

    Args:
        initial_prompt (str): The current classification prompt
        output_format_prompt (str): The output format instructions
        false_positives (list): Texts incorrectly classified as positive
        false_negatives (list): Texts incorrectly classified as negative
        true_positives (list): Texts correctly classified as positive
        invalid_outputs (list): Outputs that didn't follow the output schema
        previous_metrics (dict): Metrics from the previous iteration
        log_dir (str): Directory for storing logs
        iteration (int): Current iteration number

    Returns:
        str: The updated prompt
    """
    
    print("\nAnalyzing misclassifications, true positives, and invalid outputs...")
    
    total_predictions = previous_metrics['total_predictions']
    
    # Analyze False Positives
    num_fp = len(false_positives)
    if num_fp > 0:
        fp_texts = "\n".join(f"- {item['text']}" for item in false_positives)
        fp_percentage = (num_fp / total_predictions) * 100
        fp_fn_ratio = num_fp / len(false_negatives) if len(false_negatives) > 0 else float('inf')
        fp_analysis = get_analysis(config.FALSE_POSITIVES_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            fp_texts=fp_texts,
            total_predictions=total_predictions,
            num_fp=num_fp,
            fp_percentage=fp_percentage,
            fp_fn_ratio=fp_fn_ratio
        ))
    else:
        fp_analysis = "No false positives found in this iteration."
    display_analysis(fp_analysis, "False Positives Analysis")

    # Analyze False Negatives
    num_fn = len(false_negatives)
    if num_fn > 0:
        fn_texts = "\n".join(f"- {item['text']}" for item in false_negatives)
        fn_percentage = (num_fn / total_predictions) * 100
        fn_fp_ratio = num_fn / num_fp if num_fp > 0 else float('inf')
        fn_analysis = get_analysis(config.FALSE_NEGATIVES_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            fn_texts=fn_texts,
            total_predictions=total_predictions,
            num_fn=num_fn,
            fn_percentage=fn_percentage,
            fn_fp_ratio=fn_fp_ratio
        ))
    else:
        fn_analysis = "No false negatives found in this iteration."
    display_analysis(fn_analysis, "False Negatives Analysis")

    # Analyze True Positives
    num_tp = len(true_positives)
    if num_tp > 0:
        tp_texts = "\n".join(f"- {item['text']}" for item in true_positives)
        tp_percentage = (num_tp / total_predictions) * 100
        tp_fp_ratio = num_tp / num_fp if num_fp > 0 else float('inf')
        tp_fn_ratio = num_tp / num_fn if num_fn > 0 else float('inf')
        tp_analysis = get_analysis(config.TRUE_POSITIVES_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            tp_texts=tp_texts,
            total_predictions=total_predictions,
            num_tp=num_tp,
            tp_percentage=tp_percentage,
            tp_fp_ratio=tp_fp_ratio,
            tp_fn_ratio=tp_fn_ratio
        ))
    else:
        tp_analysis = "No true positives found in this iteration."
    display_analysis(tp_analysis, "True Positives Analysis")

    # Analyze Invalid Outputs
    num_invalid = len(invalid_outputs)
    if num_invalid > 0:
        invalid_texts = "\n".join(f"- Text: {item['text']}\n  Raw Output: {item['raw_output']}" for item in invalid_outputs)
        invalid_percentage = (num_invalid / total_predictions) * 100
        invalid_analysis = get_analysis(config.INVALID_OUTPUTS_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            output_format_prompt=output_format_prompt,
            invalid_texts=invalid_texts,
            total_predictions=total_predictions,
            num_invalid=num_invalid,
            invalid_percentage=invalid_percentage
        ))
    else:
        invalid_analysis = "No invalid outputs found in this iteration."
    display_analysis(invalid_analysis, "Invalid Outputs Analysis")

    # Generate improved prompt
    new_prompt = get_analysis(config.PROMPT_ENGINEER_INPUT.format(
        initial_prompt=initial_prompt,
        output_format_prompt=output_format_prompt,
        fp_analysis=fp_analysis,
        fn_analysis=fn_analysis,
        tp_analysis=tp_analysis,
        invalid_analysis=invalid_analysis,
        precision=previous_metrics['precision'],
        recall=previous_metrics['recall'],
        accuracy=previous_metrics['accuracy'],
        f1=previous_metrics['f1'],
        total_predictions=total_predictions,
        valid_predictions=previous_metrics['valid_predictions'],
        invalid_predictions=previous_metrics['invalid_predictions']
    ))
    
    # Log prompt generation process
    if log_dir and iteration:
        log_prompt_generation(log_dir, iteration, initial_prompt, fp_analysis, fn_analysis, tp_analysis, invalid_analysis, new_prompt)

    return new_prompt