import os
from dotenv import load_dotenv
from src.pipeline import KnowledgeGraphPipeline

def main():
    # Print current working directory and .env file existence
    print("\n--- Debug Information ---")
    print(f"Current working directory: {os.getcwd()}")
    print(f".env file exists: {os.path.exists('.env')}")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Display loaded environment variables with more detail
    print("\n--- Environment Variables Loaded ---")
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    llm_provider = os.getenv('LLM_PROVIDER', 'openai')
    print(f"LLM Provider: {llm_provider}")
    print(f"OPENAI_API_KEY: {openai_key if openai_key else 'Not Set'}")
    print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE', 'Not Set')}")
    print(f"ANTHROPIC_API_KEY: {anthropic_key if anthropic_key else 'Not Set'}")
    print("-" * 40)
    
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
    
    # Initialize and run the pipeline
    pipeline = KnowledgeGraphPipeline()
    success, result, error = pipeline.process_text(text)
    
    if success:
        pipeline.display_results(result)
    else:
        print(f"Error processing text: {error}")

if __name__ == "__main__":
    main() 