import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.evaluation.evaluator import PipelineEvaluator, EvaluationConfig
from src.config.settings import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT_TEMPLATE

def main():
    print("\n=== LLM Provider Comparison Example ===")
    print("This example compares the outputs of Anthropic and OpenAI on the same input text.")
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    if not anthropic_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        return
    
    print("✅ Both API keys found")
    
    # Example text (Marie Curie biography)
    text = """
Marie Curie, born Maria Skłodowska in Warsaw, Poland, was a pioneering physicist and chemist.
She conducted groundbreaking research on radioactivity. Together with her husband, Pierre Curie,
she discovered the elements polonium and radium. Marie Curie was the first woman to win a Nobel Prize,
the first person and only woman to win the Nobel Prize twice, and the only person to win the Nobel Prize
in two different scientific fields. She won the Nobel Prize in Physics in 1903 with Pierre Curie
and Henri Becquerel. Later, she won the Nobel Prize in Chemistry in 1911 for her work on radium and
polonium. During World War I, she developed mobile radiography units, known as 'petites Curies',
to provide X-ray services to field hospitals. Marie Curie died in 1934 from aplastic anemia, likely
caused by her long-term exposure to radiation.

Marie was born on November 7, 1867, to a family of teachers who valued education. She received her
early schooling in Warsaw but moved to Paris in 1891 to continue her studies at the Sorbonne, where
she earned degrees in physics and mathematics. She met Pierre Curie, a professor of physics, in 1894, 
and they married in 1895, beginning a productive scientific partnership. Following Pierre's tragic 
death in a street accident in 1906, Marie took over his teaching position, becoming the first female 
professor at the Sorbonne.

The Curies' work on radioactivity was conducted in challenging conditions, in a poorly equipped shed 
with no proper ventilation, as they processed tons of pitchblende ore to isolate radium. Marie Curie
established the Curie Institute in Paris, which became a major center for medical research. She had
two daughters: Irène, who later won a Nobel Prize in Chemistry with her husband, and Eve, who became
a writer. Marie's notebooks are still radioactive today and are kept in lead-lined boxes. Her legacy
includes not only her scientific discoveries but also her role in breaking gender barriers in academia
and science.
    """
    
    print(f"\n📝 Input text length: {len(text)} characters")
    
    # Create evaluation configurations for both providers
    configs = [
        EvaluationConfig(
            llm_provider="openai",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=EXTRACTION_USER_PROMPT_TEMPLATE,
            chunk_size=2000,
            chunk_overlap=100,
            input_text=text,
            temperature=0.0,
            max_tokens=4096,
            model_name="gpt-4-turbo"
        ),
        EvaluationConfig(
            llm_provider="anthropic",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=EXTRACTION_USER_PROMPT_TEMPLATE,
            chunk_size=2000,
            chunk_overlap=100,
            input_text=text,
            temperature=0.0,
            max_tokens=4096,
            model_name="claude-3-5-sonnet-20241022"
        )
    ]
    
    # Initialize evaluator
    evaluator = PipelineEvaluator(output_dir="evaluation_results")
    
    print("\n🔄 Running evaluations for both providers...")
    
    # Run individual evaluations
    results = {}
    for config in configs:
        print(f"\n--- Evaluating {config.llm_provider.upper()} ---")
        result = evaluator.evaluate_config(config)
        results[config.llm_provider] = result
        
        if result["success"]:
            stats = result["results"]["statistics"]
            print(f"✅ Success! Processed {stats['total_triples']} triples in {result['processing_time']:.2f}s")
            print(f"   Unique triples: {stats['unique_triples']}")
            print(f"   Success rate: {stats['processed_chunks']}/{stats['total_chunks']} chunks")
        else:
            print(f"❌ Failed: {result['error']}")
    
    # Compare configurations
    print(f"\n📊 Comparing configurations...")
    comparison_df = evaluator.compare_configurations(configs)
    
    if not comparison_df.empty:
        print("\n--- Comparison Results ---")
        print(comparison_df.to_string(index=False))
        
        # Save results
        evaluator.save_results("llm_comparison_results.json")
        print(f"\n💾 Results saved to evaluation_results/llm_comparison_results.json")
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        evaluator.plot_comparison(comparison_df, "processing_time")
        evaluator.plot_comparison(comparison_df, "total_triples")
        evaluator.plot_comparison(comparison_df, "unique_triples")
        print("📊 Plots saved to evaluation_results/")
    
    # Generate diff between outputs
    print(f"\n🔍 Generating diff between outputs...")
    diff_output = evaluator.compare_triples(configs[0], configs[1])
    
    if diff_output and not diff_output.startswith("Error"):
        print("\n--- Diff Output (OpenAI vs Anthropic) ---")
        print(diff_output)
        
        # Save diff to file
        diff_file = evaluator.output_dir / "llm_comparison_diff.txt"
        with open(diff_file, "w") as f:
            f.write(diff_output)
        print(f"\n💾 Diff saved to {diff_file}")
    else:
        print(f"❌ Could not generate diff: {diff_output}")
    
    # Display detailed results
    print(f"\n📋 Detailed Results Summary:")
    for provider, result in results.items():
        print(f"\n--- {provider.upper()} Results ---")
        if result["success"]:
            triples = result["results"]["triples"]
            print(f"Total triples extracted: {len(triples)}")
            
            # Show first 5 triples as examples
            print("Sample triples:")
            for i, triple in enumerate(triples[:5]):
                print(f"  {i+1}. {triple['subject']} | {triple['predicate']} | {triple['object']}")
            
            if len(triples) > 5:
                print(f"  ... and {len(triples) - 5} more")
        else:
            print(f"Failed: {result['error']}")

if __name__ == "__main__":
    main() 