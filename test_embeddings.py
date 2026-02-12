from src.embeddings import EmbeddingProvider

def test_embeddings():
    print("Testing HuggingFace Embeddings...")
    try:
        provider = EmbeddingProvider(provider="huggingface")
        text = "Hello world"
        vector = provider.embed_query(text)
        print(f"✅ Success! Vector length: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_embeddings()
