from . import config
from .model_interface import get_analysis
from .utils import display_analysis, log_prompt_generation, display_prompt

def generate_new_prompt(initial_prompt: str, output_format_prompt: str, false_positives: list, false_negatives: list, true_positives: list, invalid_outputs: list, previous_metrics: dict, log_dir: str = None, iteration: int = None, provider: str = None, model: str = None, temperature: float = 0.9, fp_comments: str = "", fn_comments: str = "", tp_comments: str = "", invalid_comments: str = "", prompt_engineering_comments: str = "") -> tuple:
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
        provider (str): Provider for the model
        model (str): Model name
        temperature (float): Temperature for the model
        fp_comments (str): Comments for false positives analysis
        fn_comments (str): Comments for false negatives analysis
        tp_comments (str): Comments for true positives analysis
        invalid_comments (str): Comments for invalid outputs analysis
        prompt_engineering_comments (str): Additional comments for prompt engineering

    Returns:
        tuple: A tuple containing the updated prompt, analysis results, and prompts used
    """
    
    print("\nAnalyzing misclassifications, true positives, and invalid outputs...")
    
    total_predictions = previous_metrics['total_predictions']
    
    analyses = {}
    prompts_used = {}

    # Analyze False Positives
    num_fp = len(false_positives)
    if num_fp > 0:
        fp_texts_and_cot = "\n\n".join(f"** Text {i+1}:\n{item['text']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n" for i, item in enumerate(false_positives))
        fp_percentage = (num_fp / total_predictions) * 100
        fp_fn_ratio = num_fp / len(false_negatives) if len(false_negatives) > 0 else float('inf')
        fp_prompt = config.FALSE_POSITIVES_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            fp_texts_and_cot=fp_texts_and_cot,
            total_predictions=total_predictions,
            num_fp=num_fp,
            fp_percentage=fp_percentage,
            fp_fn_ratio=fp_fn_ratio,
            fp_comments=fp_comments
        )
        fp_analysis = get_analysis(provider, model, temperature, fp_prompt)
    else:
        fp_analysis = "No false positives found in this iteration."
        fp_prompt = "No prompt used (no false positives)"
    display_analysis(fp_analysis, "False Positives Analysis")
    analyses['fp_analysis'] = fp_analysis
    prompts_used['fp_prompt'] = fp_prompt

    # Analyze False Negatives
    num_fn = len(false_negatives)
    if num_fn > 0:
        fn_texts_and_cot = "\n\n".join(f"** Text {i+1}:\n{item['text']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n" for i, item in enumerate(false_negatives))
        fn_percentage = (num_fn / total_predictions) * 100
        fn_fp_ratio = num_fn / num_fp if num_fp > 0 else float('inf')
        fn_prompt = config.FALSE_NEGATIVES_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            fn_texts_and_cot=fn_texts_and_cot,
            total_predictions=total_predictions,
            num_fn=num_fn,
            fn_percentage=fn_percentage,
            fn_fp_ratio=fn_fp_ratio,
            fn_comments=fn_comments
        )
        fn_analysis = get_analysis(provider, model, temperature, fn_prompt)
    else:
        fn_analysis = "No false negatives found in this iteration."
        fn_prompt = "No prompt used (no false negatives)"
    display_analysis(fn_analysis, "False Negatives Analysis")
    analyses['fn_analysis'] = fn_analysis
    prompts_used['fn_prompt'] = fn_prompt

    # Analyze True Positives
    num_tp = len(true_positives)
    if num_tp > 0:
        tp_texts_and_cot = "\n\n".join(f"** Text {i+1}:\n{item['text']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n" for i, item in enumerate(true_positives))
        tp_percentage = (num_tp / total_predictions) * 100
        tp_fp_ratio = num_tp / num_fp if num_fp > 0 else float('inf')
        tp_fn_ratio = num_tp / num_fn if num_fn > 0 else float('inf')
        tp_prompt = config.TRUE_POSITIVES_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            tp_texts_and_cot=tp_texts_and_cot,
            total_predictions=total_predictions,
            num_tp=num_tp,
            tp_percentage=tp_percentage,
            tp_fp_ratio=tp_fp_ratio,
            tp_fn_ratio=tp_fn_ratio,
            tp_comments=tp_comments
        )
        tp_analysis = get_analysis(provider, model, temperature, tp_prompt)
    else:
        tp_analysis = "No true positives found in this iteration."
        tp_prompt = "No prompt used (no true positives)"
    display_analysis(tp_analysis, "True Positives Analysis")
    analyses['tp_analysis'] = tp_analysis
    prompts_used['tp_prompt'] = tp_prompt

    # Analyze Invalid Outputs
    num_invalid = len(invalid_outputs)
    if num_invalid > 0:
        invalid_texts = "\n\n".join(f"** Text {i+1}:\n{item['text']}\n\nRaw Output:\n{item['raw_output']}\n" for i, item in enumerate(invalid_outputs))
        invalid_percentage = (num_invalid / total_predictions) * 100
        invalid_prompt = config.INVALID_OUTPUTS_ANALYSIS_PROMPT.format(
            initial_prompt=initial_prompt,
            output_format_prompt=output_format_prompt,
            invalid_texts=invalid_texts,
            total_predictions=total_predictions,
            num_invalid=num_invalid,
            invalid_percentage=invalid_percentage,
            invalid_comments=invalid_comments
        )
        invalid_analysis = get_analysis(provider, model, temperature, invalid_prompt)
    else:
        invalid_analysis = "No invalid outputs found in this iteration."
        invalid_prompt = "No prompt used (no invalid outputs)"
    display_analysis(invalid_analysis, "Invalid Outputs Analysis")
    analyses['invalid_analysis'] = invalid_analysis
    prompts_used['invalid_prompt'] = invalid_prompt

    # Generate improved prompt
    prompt_engineer_input = config.PROMPT_ENGINEER_INPUT.format(
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
        invalid_predictions=previous_metrics['invalid_predictions'],
        prompt_engineering_comments=prompt_engineering_comments
    )
    new_prompt = get_analysis(provider, model, temperature, prompt_engineer_input)
    prompts_used['prompt_engineer_input'] = prompt_engineer_input
    
    # Log prompt generation process with prompts used
    if log_dir and iteration:
        log_prompt_generation(
            log_dir, 
            iteration, 
            initial_prompt, 
            fp_analysis, 
            fn_analysis, 
            tp_analysis, 
            invalid_analysis, 
            new_prompt,
            prompts_used
        )

    return new_prompt, analyses, prompts_used

def validate_and_improve_prompt(new_prompt: str, output_format_prompt: str, provider: str = None, model: str = None, temperature: float = 0.9, validation_comments: str = "") -> tuple:
    """
    Validates the new prompt against best practices and improves it if necessary.

    Args:
        new_prompt (str): The newly generated prompt
        output_format_prompt (str): The output format instructions
        provider (str): Provider for the model
        model (str): Model name
        temperature (float): Temperature for the model
        validation_comments (str): Comments for validation and improvement

    Returns:
        tuple: A tuple containing the improved prompt and the validation result
    """
    print("\nValidating and improving the new prompt...")

    validation_prompt = config.VALIDATION_AND_IMPROVEMENT_PROMPT.format(
        new_prompt=new_prompt,
        output_format_prompt=output_format_prompt,
        validation_comments=validation_comments
    )

    validation_result = get_analysis(provider, model, temperature, validation_prompt)
    
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
    
    return improved_prompt, validation_result
