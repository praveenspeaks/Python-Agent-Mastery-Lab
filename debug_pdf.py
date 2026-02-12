from src.parser import DocumentParser
import os

def final_debug():
    file_path = r"d:\My Professional Projects\PythonAgent\SMB_Speed_Enterprise_Control.pdf"
    parser = DocumentParser()
    try:
        print(f"Testing official parser on: {file_path}")
        docs = parser.load_document(file_path)
        print("Final Status: SUCCESS")
    except Exception as e:
        print(f"\nFinal Status: FAILED")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    final_debug()
