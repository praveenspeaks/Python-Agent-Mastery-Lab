from src.app import LangChainAgent
import os

def run_embedding_demo():
    print("🚀 **LangChain Chapter 2: Embeddings & Vector Stores Demo**")
    print("-" * 50)
    
    agent = LangChainAgent()
    
    # 1. Indexing
    test_file = "testdata/sample_text.txt"
    if not os.path.exists(test_file):
        print(f"Creating {test_file}...")
        os.makedirs("testdata", exist_ok=True)
        with open(test_file, "w") as f:
            f.write("Artificial intelligence (AI) is intelligence demonstrated by machines.\n")
            f.write("Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.\n")
            f.write("Deep learning is part of a broader family of machine learning methods based on artificial neural networks.\n")
            f.write("Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.\n")

    logs = agent.index_file(test_file)
    for log in logs:
        print(log["message"])
    
    # 2. Semantic Search
    queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "How do computers understand language?"
    ]
    
    print("\n🔍 **Semantic Search Results:**")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results, logs = agent.search_documents(query, k=2)
        for i, doc in enumerate(results):
            print(f"  [{i+1}] {doc.page_content[:100]}...")

if __name__ == "__main__":
    run_embedding_demo()
