from . import config
from .model_interface import get_analysis
from .utils import display_analysis, log_prompt_generation

def generate_new_prompt(initial_prompt: str, output_format_prompt: str, false_positives: list, false_negatives: list, previous_metrics: dict, log_dir: str = None, iteration: int = None) -> str:
    """
    Generates a new prompt by incorporating false positives and false negatives.

    Args:
        initial_prompt (str): The initial classification prompt
        output_format_prompt (str): The output format instructions
        false_positives (list): Texts incorrectly classified as positive
        false_negatives (list): Texts incorrectly classified as negative
        previous_metrics (dict): Metrics from the previous iteration
        log_dir (str): Directory for storing logs
        iteration (int): Current iteration number

    Returns:
        str: The updated prompt
    """
    
    print("\nAnalyzing misclassifications...")
    fp_texts = "\n".join(f"- {item['text']}" for item in false_positives)
    fn_texts = "\n".join(f"- {item['text']}" for item in false_negatives)

    analysis_prompt = config.ANALYSIS_PROMPT.format(fp_texts=fp_texts, fn_texts=fn_texts)

    analysis = get_analysis(analysis_prompt)
    display_analysis(analysis)

    new_prompt = generate_improved_prompt(initial_prompt, analysis, previous_metrics)
    log_prompt_generation(log_dir, iteration, initial_prompt, analysis, new_prompt)

    return new_prompt

def generate_improved_prompt(initial_prompt: str, analysis: str, previous_metrics: dict) -> str:
    """Generates an improved prompt based on the analysis and previous metrics."""
    print("\nGenerating new prompt...")
    prompt_engineer_input = config.PROMPT_ENGINEER_INPUT.format(
        initial_prompt=initial_prompt,
        analysis=analysis,
        precision=previous_metrics['precision'],
        recall=previous_metrics['recall'],
        accuracy=previous_metrics['accuracy'],
        f1_score=previous_metrics['f1']
    )

    return get_analysis(prompt_engineer_input)