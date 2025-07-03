import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.pipeline import KnowledgeGraphPipeline
from src.config.configuration import Configuration
from src.storage.jsonld_graphdb_storage import JSONLDGraphDBStorage

def main():
    print("\n--- Debug Information ---")
    print(f"Current working directory: {os.getcwd()}")
    print(f".env file exists: {os.path.exists('.env')}")
    load_dotenv()
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
    
    # Path to the ontology file (for JSON-LD extraction)
    ontology_path = "examples/scientist_ontology.owl"

    # Set up configuration for JSON-LD extraction
    config = Configuration.from_env()
    config.extraction.extraction_mode = "jsonld"
    config.extraction.ontology_path = ontology_path

    pipeline = KnowledgeGraphPipeline(config)
    pipeline.display_configuration()
    success, result, error = pipeline.process_text(text)

    if not success:
        print(f"Error processing text: {error}")
        return

    pipeline.display_results(result)
    pipeline.display_summary(result)

    # Upload JSON-LD to GraphDB
    jsonld_data = result.get('jsonld')
    if not jsonld_data:
        print("No JSON-LD data to upload.")
        return

    repo_id = os.getenv('GRAPHDB_REPO_ID', 'test-repo')
    base_url = os.getenv('GRAPHDB_BASE_URL', 'http://localhost:7200')
    storage = JSONLDGraphDBStorage(repo_id, base_url)
    success = storage.upload_jsonld(jsonld_data)
    if success:
        print("Successfully uploaded JSON-LD to GraphDB.")
    else:
        print("Failed to upload JSON-LD to GraphDB.")

if __name__ == "__main__":
    main() 