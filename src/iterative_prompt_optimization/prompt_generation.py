from .model_interface import get_analysis
from .utils import display_analysis, log_prompt_generation

def generate_new_prompt(initial_prompt: str, output_format_prompt: str, false_positives: list, false_negatives: list, log_dir: str = None, iteration: int = None) -> str:
    """
    Generate a new prompt based on the analysis of misclassifications.

    This function:
    1. Creates an analysis prompt using misclassified examples
    2. Gets an analysis from the AI model
    3. Generates an improved prompt based on the analysis
    4. Logs the prompt generation process

    Args:
        initial_prompt (str): The current prompt being used
        output_format_prompt (str): Instructions for the desired output format
        false_positives (list): List of texts incorrectly classified as positive
        false_negatives (list): List of texts incorrectly classified as negative
        log_dir (str, optional): Directory for storing logs
        iteration (int, optional): Current iteration number

    Returns:
        str: The newly generated prompt
    """
    print("\nAnalyzing misclassifications...")
    fp_texts = "\n".join(f"- {item['text']}" for item in false_positives)
    fn_texts = "\n".join(f"- {item['text']}" for item in false_negatives)

    analysis_prompt = _create_analysis_prompt(fp_texts, fn_texts)

    analysis = get_analysis(analysis_prompt)
    display_analysis(analysis)

    new_prompt = generate_improved_prompt(initial_prompt, analysis)
    log_prompt_generation(log_dir, iteration, initial_prompt, analysis, new_prompt)

    return new_prompt

def _create_analysis_prompt(fp_texts: str, fn_texts: str) -> str:
    """
    Create a prompt for analyzing misclassifications.

    Args:
        fp_texts (str): String of false positive texts
        fn_texts (str): String of false negative texts

    Returns:
        str: The analysis prompt
    """
    return f"""
    You are an expert in refining LLMs prompts for binary classifications. Below are two sets of texts that were misclassified by the LLM model:

        Negative (0) texts (incorrectly classified as positive):
        {fp_texts}

        Positives (0) texts (incorrectly classified as negative):
        {fn_texts}

    Your task is to analyze these misclassifications and provide insights into why these errors occurred. Identify specific examples from each set where the model made a mistake and highlight what elements of the text may have led to the incorrect classification. Additionally, specify what the correct classification should have been for each example.

    Based on your analysis, suggest strategies to improve the classification prompt, focusing on how it can better recognize the nuances that led to the errors. Your recommendations should include ways to reduce both false positives and false negatives by making the prompt more sensitive to subtle differences in the classification of text.
    """

def generate_improved_prompt(initial_prompt: str, analysis: str) -> str:
    """
    Generate an improved prompt based on the analysis.

    Args:
        initial_prompt (str): The current prompt being used
        analysis (str): The analysis of misclassifications

    Returns:
        str: The improved prompt
    """
    print("\nGenerating new prompt...")
    prompt_engineer_input = f"""
        You are an expert in crafting highly effective prompts. Your task is to help me improve a prompt for binary classification. I will give you the current prompt and an analysis showing where it failed to classify a piece of text correctly. Your goal is to refine the prompt to be more precise and adaptable, ensuring that the AI can accurately classify similar texts going forward. The revised prompt should be written in the first person, guiding the AI to handle difficult or edge cases.

        Current prompt:
        {initial_prompt}

        Analysis of misclassifications:
        {analysis}

        Your task is to provide a rewritten, production-ready version of the prompt that improves its accuracy. 
        
        IMPORTANT note: the prompt should not include any preamble or request for explanations, just the final prompt itself.
        """

    return get_analysis(prompt_engineer_input)