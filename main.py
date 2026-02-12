import os
from src.app import LangChainAgent

def main():
    agent = LangChainAgent(chunk_size=150, chunk_overlap=20)
    test_dir = "testdata"
    
    if not os.path.exists(test_dir):
        print(f"Directory {test_dir} not found. Please run generate_test_data.py first.")
        return

    # Get all files in the testdata directory
    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    
    print(f"Found {len(files)} files in {test_dir}. Starting batch processing...")
    
    for file_name in files:
        file_path = os.path.join(test_dir, file_name)
        agent.process_file(file_path)

if __name__ == "__main__":
    main()
