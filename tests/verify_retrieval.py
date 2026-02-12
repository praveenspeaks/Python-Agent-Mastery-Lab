import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import LangChainAgent

def test_retrieval_functions():
    print("Testing LangChainAgent Retrieval Features...")
    
    # 1. Initialize
    try:
        agent = LangChainAgent()
        print("✅ Initialization Successful.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 2. Create Dummy File
    demo_file = "test_data.txt"
    with open(demo_file, "w", encoding="utf-8") as f:
        f.write("Artificial Intelligence (AI) is intelligence demonstrated by machines.\n")
        f.write("Machine Learning (ML) is a subset of AI.\n")
        f.write("Deep Learning is a subset of ML based on artificial neural networks.")

    # 3. Process File (Ingestion)
    print("\n--- Testing File Ingestion ---")
    chunks, logs = agent.process_file(demo_file)
    if chunks and len(chunks) > 0:
        print(f"✅ Processed {len(chunks)} chunks.")
    else:
        print("❌ File processing returned no chunks.")
        for l in logs: print(l)

    # 4. Vector Store Indexing
    print("\n--- Testing Vector Indexing ---")
    logs = agent.index_file(demo_file)
    # Check logs for success
    if any("Successfully indexed" in l["message"] for l in logs):
         print("✅ Indexing Successful.")
    else:
         print("❌ Indexing might have failed.")
         for l in logs: print(l['message'])

    # 5. Search
    print("\n--- Testing Search ---")
    results, logs = agent.search_documents("What is AI?")
    if results:
        print(f"✅ Search returned {len(results)} results.")
        print(f"   Top result: {results[0].page_content[:50]}...")
    else:
        print("⚠️ Search returned no results (might be expected if embeddings are invalid/mocked).")

    # 6. HyDE Search
    print("\n--- Testing HyDE Search ---")
    # This calls the LLM, so it might fail if no key.
    # But we want to ensure the function runs without syntax error.
    hyde_doc, logs = agent.hyde_search("Explain Neural Networks")
    if hyde_doc:
        print("✅ HyDE Generation ran (mock or real).")
    else:
        print("❌ HyDE returned nothing.")

    # 7. Cleanup
    if os.path.exists(demo_file):
        os.remove(demo_file)
    
    print("\nTest Complete.")

if __name__ == "__main__":
    test_retrieval_functions()
