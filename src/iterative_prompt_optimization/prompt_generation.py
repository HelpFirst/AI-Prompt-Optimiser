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
    
    # Analyze False Positives
    fp_texts = "\n".join(f"- {item['text']}" for item in false_positives)
    fp_analysis = get_analysis(config.FALSE_POSITIVES_ANALYSIS_PROMPT.format(initial_prompt=initial_prompt, fp_texts=fp_texts))
    display_analysis(fp_analysis, "False Positives Analysis")

    # Analyze False Negatives
    fn_texts = "\n".join(f"- {item['text']}" for item in false_negatives)
    fn_analysis = get_analysis(config.FALSE_NEGATIVES_ANALYSIS_PROMPT.format(initial_prompt=initial_prompt, fn_texts=fn_texts))
    display_analysis(fn_analysis, "False Negatives Analysis")

    # Analyze True Positives
    tp_texts = "\n".join(f"- {item['text']}" for item in true_positives)
    tp_analysis = get_analysis(config.TRUE_POSITIVES_ANALYSIS_PROMPT.format(initial_prompt=initial_prompt, tp_texts=tp_texts))
    display_analysis(tp_analysis, "True Positives Analysis")

    # Analyze Invalid Outputs
    invalid_texts = "\n".join(f"- Text: {item['text']}\n  Raw Output: {item['raw_output']}" for item in invalid_outputs)
    invalid_analysis = get_analysis(config.INVALID_OUTPUTS_ANALYSIS_PROMPT.format(
        initial_prompt=initial_prompt,
        output_format_prompt=output_format_prompt,
        invalid_texts=invalid_texts
    ))
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
        f1=previous_metrics['f1']
    ))
    
    # Log prompt generation process
    if log_dir and iteration:
        log_prompt_generation(log_dir, iteration, initial_prompt, fp_analysis, fn_analysis, tp_analysis, invalid_analysis, new_prompt)

    return new_prompt