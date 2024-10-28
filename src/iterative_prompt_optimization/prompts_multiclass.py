# Template prompts for multiclass classification analysis and optimization
# These prompts guide the AI in analyzing and improving classification performance

# Prompt template for analyzing correct predictions
# Used to identify successful patterns and effective classification strategies
CORRECT_PREDICTIONS_ANALYSIS_PROMPT = """
Analyze the following correct prediction examples for multiclass classification:

Current Prompt:
{initial_prompt}

Correct Prediction Examples:
{correct_texts_and_cot}

Context:
- Total number of predictions: {total_predictions}
- Number of correct predictions: {num_correct}
- Percentage of correct predictions: {correct_percentage:.2f}%
- Distribution across classes: {class_distribution}

Additional Comments:
{correct_comments}

Provide a concise analysis of these correct predictions, focusing on:
1. Common patterns in the chain of thought that led to successful classifications
2. Key features or indicators that consistently map to specific categories
3. Clear category boundaries and distinguishing characteristics
4. Effective reasoning strategies used across different classes

Consider both the text content and the chain of thought in your analysis. Suggest brief ways to reinforce these successful patterns.

Limit your response to 3-5 key points, prioritizing the most impactful observations.
"""

# Main prompt template for improving multiclass classification prompts
# Guides the AI in optimizing the prompt based on analysis results
PROMPT_ENGINEER_INPUT_MULTICLASS = """
You are an expert in refining prompts for multiclass classification tasks.

## Goal: Optimize the current prompt based on the analysis of correct and incorrect predictions and performance metrics.

---- PROMPT TO BE IMPROVED ----

## Current Prompt:
{initial_prompt}

## Metrics:
- Overall Accuracy: {accuracy}
- Per-class Metrics:
{per_class_metrics}
- Total Predictions: {total_predictions}
- Valid Predictions: {valid_predictions}
- Invalid Predictions: {invalid_predictions}

---- ANALYSIS ----

## Correct Predictions Analysis:
{correct_analysis}

## Incorrect Predictions Analysis:
{incorrect_analysis}

---- OUTPUT FORMAT INSTRUCTIONS ----

## Output Format Instructions:
{output_format_prompt}

---- ADDITIONAL COMMENTS ----
{prompt_engineering_comments}

---- IMPROVED PROMPT ----

## GOAL: Create a concise, improved prompt that:
1. Reinforces patterns found in correct predictions
2. Addresses common misclassification patterns
3. Clarifies ambiguous category boundaries
4. Maintains clear distinctions between categories
5. Improves overall classification accuracy
6. Adheres to the output format instructions

## INSTRUCTIONS:
1. Keep the prompt clear and focused
2. Include specific guidance for distinguishing between similar categories
3. Add negative instructions to prevent common misclassifications
4. Provide the new prompt in plain text without additional explanations
5. Include step-by-step reasoning instructions for multiclass decisions

Provide only the improved prompt without any additional commentary.
"""

# Prompt template for analyzing incorrect predictions
# Used to identify error patterns and areas for improvement
INCORRECT_PREDICTIONS_ANALYSIS_PROMPT = """
Analyze the following incorrect prediction examples for multiclass classification:

Current Prompt:
{initial_prompt}

Incorrect Prediction Examples:
{incorrect_texts_and_cot}

Context:
- Total number of predictions: {total_predictions}
- Number of incorrect predictions: {num_incorrect}
- Percentage of incorrect predictions: {incorrect_percentage:.2f}%
- Distribution of misclassifications: {misclassification_distribution}

Additional Comments:
{incorrect_comments}

Provide a concise analysis of these incorrect predictions, focusing on:
1. Common patterns in misclassifications between specific categories
2. Ambiguous elements that led to incorrect classifications
3. Chain of thought errors or misleading reasoning patterns
4. Potential category boundary confusion points

For each key observation:
- Identify the source of confusion
- Note any recurring patterns in the chain of thought
- Suggest specific improvements to prevent similar errors

Consider both the text content and the chain of thought in your analysis. 
Prioritize the most impactful patterns that could improve classification accuracy.

Limit your response to 3-5 key points, focusing on the most significant error patterns.
"""
