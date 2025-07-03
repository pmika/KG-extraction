from src.evaluation.evaluator import PipelineEvaluator, EvaluationConfig
from src.config.settings import EXTRACTION_SYSTEM_PROMPT
from src.config.settings import EXTRACTION_USER_PROMPT_TEMPLATE
from src.config.settings import LLM_MODEL_NAMES

def main():
    # Sample text to process
    sample_text = """
    Albert Einstein was born in Germany in 1879. He developed the theory of relativity.
    Marie Curie was a Polish physicist who conducted pioneering research on radioactivity.
    She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different fields.
    """

    # Create two different configurations to compare
    config1 = EvaluationConfig(
        llm_provider="openai",
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=EXTRACTION_USER_PROMPT_TEMPLATE,
        chunk_size=1000,
        chunk_overlap=200,
        input_text=sample_text,
        temperature=0.7,
        model_name=LLM_MODEL_NAMES["openai"]
    )

    config2 = EvaluationConfig(
        llm_provider="anthropic",
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=EXTRACTION_USER_PROMPT_TEMPLATE,
        chunk_size=800,
        chunk_overlap=100,
        input_text=sample_text,
        temperature=0.5,
        model_name=LLM_MODEL_NAMES["anthropic"]
    )

    # Initialize the evaluator
    evaluator = PipelineEvaluator(output_dir="evaluation_results")

    # Compare the triples from both configurations
    print("Comparing triples between configurations...")
    diff_output = evaluator.compare_triples(config1, config2)
    print("\nDiff Output:")
    print(diff_output)

    # Save the evaluation results
    evaluator.save_results("triple_comparison_results.json")

if __name__ == "__main__":
    main() 