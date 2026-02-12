import os
from src.app import LangChainAgent
from dotenv import load_dotenv

def main():
    # 1. Load your .env file
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or "your_openrouter_api_key" in api_key:
        print("\n[!] Please set your OPENROUTER_API_KEY in the .env file first.")
        return

    # 2. Initialize the agent
    # You can specify a custom model here if you want:
    # agent = LangChainAgent(model_name="anthropic/claude-3-opus")
    agent = LangChainAgent()

    # 3. Process a file with AI Summarization enabled
    test_file = "testdata/sample_text.txt"
    
    if os.path.exists(test_file):
        print(f"Starting AI analysis for {test_file}...")
        agent.process_file(test_file, summarize=True)
        
        # 4. Ask a specific question about the document
        print("\n--- Asking AI a specific question ---")
        chunks = agent.parser.load_document(test_file)
        answer = agent.ai.ask_about_document(
            "What is the main topic of this document and what is said about LLMs?",
            chunks
        )
        print(f"Question: What is the main topic of this document and what is said about LLMs?")
        print(f"AI Answer: {answer}")
    else:
        print(f"Test file {test_file} not found. Please run generate_test_data.py first.")

if __name__ == "__main__":
    main()
