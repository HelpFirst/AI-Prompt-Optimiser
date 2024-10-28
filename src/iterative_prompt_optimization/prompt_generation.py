# Module for generating and improving prompts for binary classification
# Handles analysis of prediction patterns and prompt optimization

from . import config
from .model_interface import get_analysis
from .utils import display_analysis, log_prompt_generation, display_prompt

def generate_new_prompt(initial_prompt: str, output_format_prompt: str, 
                       false_positives: list, false_negatives: list, 
                       true_positives: list, invalid_outputs: list, 
                       previous_metrics: dict, log_dir: str = None, 
                       iteration: int = None, provider: str = None, 
                       model: str = None, temperature: float = 0.9, 
                       fp_comments: str = "", fn_comments: str = "", 
                       tp_comments: str = "", invalid_comments: str = "", 
                       prompt_engineering_comments: str = "") -> tuple:
    """
    Generates a new prompt by analyzing different types of predictions.
    
    This function:
    1. Analyzes false positives to identify overclassification patterns
    2. Analyzes false negatives to identify missed classifications
    3. Analyzes true positives to identify successful patterns
    4. Analyzes invalid outputs to improve format adherence
    5. Combines analyses to generate an improved prompt
    
    Args:
        initial_prompt: Starting prompt to be improved
        output_format_prompt: Instructions for output formatting
        false_positives: List of incorrectly positive predictions
        false_negatives: List of incorrectly negative predictions
        true_positives: List of correctly positive predictions
        invalid_outputs: List of malformed outputs
        previous_metrics: Performance metrics from previous iteration
        log_dir: Directory for storing logs
        iteration: Current iteration number
        provider: AI provider for analysis
        model: Model name for analysis
        temperature: Temperature setting for generation
        *_comments: Additional guidance for different analyses
        
    Returns:
        tuple: (new_prompt, analyses_dict, prompts_used_dict)
    """
    print("\nAnalyzing misclassifications, true positives, and invalid outputs...")
    
    total_predictions = previous_metrics['total_predictions']
    analyses = {}
    prompts_used = {}

    # Analyze False Positives (overclassification errors)
    num_fp = len(false_positives)
    if num_fp > 0:
        # Format examples with their chain of thought
        fp_texts_and_cot = "\n\n".join(
            f"** Text {i+1}:\n{item['text']}\n\nChain of Thought:\n{item.get('chain_of_thought', 'N/A')}\n" 
            for i, item in enumerate(false_positives)
        )
        fp_percentage = (num_fp / total_predictions) * 100
        fp_fn_ratio = num_fp / len(false_negatives) if len(false_negatives) > 0 else float('inf')
        
        # Generate analysis prompt for false positives
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

    # Similar analysis blocks for false negatives, true positives, and invalid outputs
    # Each block follows the same pattern:
    # 1. Check if there are examples to analyze
    # 2. Format the examples with relevant information
    # 3. Calculate statistics
    # 4. Generate and get analysis
    # 5. Store results
    
    # Generate improved prompt using all analyses
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
    
    # Get the new improved prompt
    new_prompt = get_analysis(provider, model, temperature, prompt_engineer_input)
    prompts_used['prompt_engineer_input'] = prompt_engineer_input
    
    # Log the prompt generation process if enabled
    if log_dir and iteration:
        log_prompt_generation(log_dir, iteration, initial_prompt, 
                            fp_analysis, fn_analysis, tp_analysis, 
                            invalid_analysis, new_prompt)

    return new_prompt, analyses, prompts_used

def validate_and_improve_prompt(new_prompt: str, output_format_prompt: str, 
                              provider: str = None, model: str = None, 
                              temperature: float = 0.9, 
                              validation_comments: str = "") -> tuple:
    """
    Validates and potentially improves a newly generated prompt.
    
    This function:
    1. Checks if the prompt follows best practices
    2. Verifies output format compatibility
    3. Suggests improvements if needed
    
    Args:
        new_prompt: Prompt to validate
        output_format_prompt: Required output format
        provider: AI provider for validation
        model: Model name for validation
        temperature: Temperature setting
        validation_comments: Additional validation guidance
        
    Returns:
        tuple: (improved_prompt, validation_result)
    """
    print("\nValidating and improving the new prompt...")

    # Generate validation prompt
    validation_prompt = config.VALIDATION_AND_IMPROVEMENT_PROMPT.format(
        new_prompt=new_prompt,
        output_format_prompt=output_format_prompt,
        validation_comments=validation_comments
    )

    # Get validation analysis
    validation_result = get_analysis(provider, model, temperature, validation_prompt)
    
    # Parse validation result
    validation_status = "VALID" if "Validation: VALID" in validation_result else "NEEDS IMPROVEMENT"
    improved_prompt = new_prompt  # Default to original prompt
    
    if validation_status == "NEEDS IMPROVEMENT":
        # Extract improved prompt from validation result
        start_index = validation_result.find("Improved Prompt:") + len("Improved Prompt:")
        end_index = validation_result.find("Explanation:")
        improved_prompt = validation_result[start_index:end_index].strip()
    
    # Display results
    display_prompt(new_prompt, "Original Prompt")
    display_analysis(validation_result, "Prompt Validation and Improvement")
    if improved_prompt != new_prompt:
        display_prompt(improved_prompt, "Improved Prompt")
    else:
        print("No improvements were necessary. The original prompt is valid.")
    
    return improved_prompt, validation_result
