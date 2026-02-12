try:
    from langchain_core.memory import BaseMemory
    print("✅ Successfully imported langchain_core.memory.BaseMemory")
except ImportError as e:
    print(f"❌ Failed to import langchain_core.memory.BaseMemory: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
