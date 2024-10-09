PROMPT_ENGINEER_INPUT = """
You are an expert in refining prompts for classification tasks. 

## Goal: Your goal is to optimize the current prompt. Based on a given prompt, and  analyses and metrics, generate an improved prompt. You will be provided with the current prompt (to be improved), the output format instructions (to ensure that the improved prompt will generate outputs in the correct format), and the analysis of the false positive (to include suggestion in the new prompt to avoid these errors), false negatives (to include suggestion in the new prompt to avoid these errors), true positives (to include suggestion in the new prompt to reinforce these patters) and invalid outputs (to include suggestion in the new prompt to avoid these output formatting errors).

---- PROMPT TO BE IMPROVED ----

## Current Prompt (prompt to be improved):
{initial_prompt}

## Metrics (performance of the current prompt):
- Precision: {precision}
- Recall: {recall}
- Accuracy: {accuracy}
- F1 Score: {f1}
- Total Predictions: {total_predictions}
- Valid Predictions: {valid_predictions}
- Invalid Predictions (wrong output formatting): {invalid_predictions}

---- ANALYSIS ----

## True Positives Analysis: (to include suggestion in the new prompt to reinforce these patters)
{tp_analysis}

## False Positives Analysis (to include suggestion in the new prompt to avoid these errors):
{fp_analysis}

## False Negatives Analysis (to include suggestion in the new prompt to avoid these errors)::
{fn_analysis}

## Invalid Outputs Analysis (to include suggestion in the new prompt to avoid these output formatting errors):
{invalid_analysis}

---- OUTPUT FORMAT INSTRUCTIONS ----

## Output Format Instructions (to ensure that the improved prompt will generate outputs in the correct format):
{output_format_prompt}

---- IMPROVED PROMPT ----

## GOAL: Create a concise, improved prompt that addresses the most critical issues identified in the analyses. Focus on:
1. Increasing precision and recall of the classification task
2. Maintaining or improving overall accuracy and F1 score
3. Reducing invalid outputs
4. Adhering to the output format instructions

## INSTRUCTIONS:
1. Keep the prompt short and focused.
2. Prioritize clarity and effectiveness over length.
3. Provide only the new, improved prompt in plain text without any additional explanations or headers.
4. Include the chain of thought in the prompt to guide the model to think step by step with custom instructions.

## GENERARAL TACTICS FOR PROMPT WRITTING:
1. Include details in your query to get more relevant answers
2. Ask the model to adopt a persona
3. Use delimiters to clearly indicate distinct parts of the input
4.Specify the steps required to complete a task
5.Provide examples
6. Specify the desired length of the output

"""

# Analysis prompts
FALSE_POSITIVES_ANALYSIS_PROMPT = """
Analyze the following false positive examples for the given prompt:

Current Prompt:
{initial_prompt}

False Positive Examples:
{fp_texts_and_cot}

Context:
- Total number of predictions: {total_predictions}
- Number of false positives: {num_fp}
- Percentage of false positives: {fp_percentage:.2f}%
- False positive to false negative ratio: {fp_fn_ratio:.2f}

Additional Comments:
{fp_comments}

Provide a concise analysis of these false positives, focusing only on high-confidence observations. Consider both the text and the chain of thought in your analysis. Suggest brief, impactful improvements to increase precision. Prioritize the most critical issues.

Limit your response to 3-5 key points.
"""

FALSE_NEGATIVES_ANALYSIS_PROMPT = """
Analyze the following false negative examples for the given prompt:

Current Prompt:
{initial_prompt}

False Negative Examples:
{fn_texts_and_cot}

Context:
- Total number of predictions: {total_predictions}
- Number of false negatives: {num_fn}
- Percentage of false negatives: {fn_percentage:.2f}%
- False negative to false positive ratio: {fn_fp_ratio:.2f}

Additional Comments:
{fn_comments}

Provide a concise analysis of these false negatives, focusing only on high-confidence observations. Consider both the text and the chain of thought in your analysis. Suggest brief, impactful improvements to increase recall. Prioritize the most critical issues.

Limit your response to 3-5 key points.
"""

TRUE_POSITIVES_ANALYSIS_PROMPT = """
Analyze the following true positive examples for the given prompt:

Current Prompt:
{initial_prompt}

True Positive Examples:
{tp_texts_and_cot}

Context:
- Total number of predictions: {total_predictions}
- Number of true positives: {num_tp}
- Percentage of true positives: {tp_percentage:.2f}%
- True positive to false positive ratio: {tp_fp_ratio:.2f}
- True positive to false negative ratio: {tp_fn_ratio:.2f}

Additional Comments:
{tp_comments}

Provide a concise analysis of these true positives, focusing only on high-confidence observations. Consider both the text and the chain of thought in your analysis. Identify key elements that lead to successful classifications. Suggest brief ways to reinforce these patterns.

Limit your response to 3-5 key points.
"""

# Add this new prompt for analyzing invalid outputs
INVALID_OUTPUTS_ANALYSIS_PROMPT = """
Analyze the following invalid outputs for the given prompt and output format instructions:

Current Prompt:
{initial_prompt}

Output Format Instructions:
{output_format_prompt}

Invalid Outputs:
{invalid_texts}

Context:
- Total number of predictions: {total_predictions}
- Number of invalid outputs: {num_invalid}
- Percentage of invalid outputs: {invalid_percentage:.2f}%

Additional Comments:
{invalid_comments}

Provide a concise analysis of these invalid outputs, focusing only on high-confidence observations. Suggest brief, impactful improvements to reduce formatting errors and ensure adherence to the specified output format.

Important:
1. Do NOT suggest changes to the output schema or format itself.
2. Do NOT propose new keys, variables, or alterations to the existing dictionary structure.
3. Focus solely on improving the prompt to achieve the current output format more consistently.

Your suggestions should aim to guide the model to produce outputs that match the existing schema more accurately.

Limit your response to 3-5 key points, prioritizing the most critical issues that lead to invalid outputs.
"""

# Add this new prompt for validation and improvement
VALIDATION_AND_IMPROVEMENT_PROMPT = """
As an expert in prompt engineering, your task is to validate and improve the following prompt:

Current Prompt:
{new_prompt}

Output Format Instructions:
{output_format_prompt}

Additional Comments:
{validation_comments}

Please evaluate this prompt based on the following best practices:
1. Clarity and specificity: Ensure the prompt is clear and specific about the classification task.
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
