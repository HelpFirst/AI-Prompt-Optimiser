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

def validate_and_improve_prompt(new_prompt: str, output_format_prompt: str) -> str:
    """
    Validates the new prompt against best practices and improves it if necessary.

    Args:
        new_prompt (str): The newly generated prompt
        output_format_prompt (str): The output format instructions

    Returns:
        str: The validated and potentially improved prompt
    """
    print("\nValidating and improving the new prompt...")

    validation_prompt = f"""
    As an expert in prompt engineering, your task is to validate and improve the following prompt:

    Current Prompt:
    {new_prompt}

    Output Format Instructions:
    {output_format_prompt}

    Please evaluate this prompt based on the following best practices:
    1. Clarity and specificity: Ensure the prompt is clear and specific about the task.
    2. Contextual information: Check if the prompt provides necessary context.
    3. Explicit instructions: Verify that the prompt gives explicit instructions on how to approach the task.
    4. Examples: If applicable, check if the prompt includes relevant examples.
    5. Output format: Confirm that the prompt clearly specifies the desired output format.
    6. Avoiding biases: Ensure the prompt doesn't introduce unintended biases.
    7. Appropriate length: Check if the prompt is concise yet comprehensive.
    8. Task-specific considerations: Ensure the prompt addresses any specific requirements of the classification task.

    If the prompt adheres to these best practices, please confirm its validity. If improvements are needed, please provide an enhanced version of the prompt that addresses any shortcomings while maintaining its original intent and compatibility with the output format instructions.

    Your response should be in the following format:
    Validation: [VALID/NEEDS IMPROVEMENT]
    Improved Prompt: [If NEEDS IMPROVEMENT, provide the improved prompt here. If VALID, repeat the original prompt.]
    Explanation: [Brief explanation of your assessment and any changes made]
    """

    validation_result = get_analysis(validation_prompt)
    
    # Parse the validation result
    validation_status = "VALID" if "Validation: VALID" in validation_result else "NEEDS IMPROVEMENT"
    improved_prompt = new_prompt  # Default to the original prompt
    
    if validation_status == "NEEDS IMPROVEMENT":
        # Extract the improved prompt from the validation result
        start_index = validation_result.find("Improved Prompt:") + len("Improved Prompt:")
        end_index = validation_result.find("Explanation:")
        improved_prompt = validation_result[start_index:end_index].strip()
    
    # Display the prompt before and after improvement
    display_prompt(new_prompt, "Original Prompt")
    # Display the validation result
    display_analysis(validation_result, "Prompt Validation and Improvement")
    # Display the improved prompt if it's different from the original
    if improved_prompt != new_prompt:
        display_prompt(improved_prompt, "Improved Prompt")
    else:
        print("No improvements were necessary. The original prompt is valid.")
    
    return improved_prompt