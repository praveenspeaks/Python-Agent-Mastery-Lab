try:
    from typing import List, Optional
    print("✅ typing imported")
    import os
    print("✅ os imported")
    from langchain_community.vectorstores import FAISS
    print("✅ FAISS imported")
    from langchain_community.retrievers import BM25Retriever
    print("✅ BM25Retriever imported")
    from langchain_core.documents import Document
    print("✅ Document imported")
    # The likely culprits
    from langchain.retrievers import EnsembleRetriever
    print("✅ EnsembleRetriever imported")
    from langchain.retrievers import ContextualCompressionRetriever
    print("✅ ContextualCompressionRetriever imported")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
except Exception as e:
    print(f"❌ Other Error: {e}")
