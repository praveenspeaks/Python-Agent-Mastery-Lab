import ast
import os

def check_syntax(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ Syntax OK: {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax Error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
files_to_check = [
    "learning_lab.py",
    "src/app.py",
    "src/ai_model.py",
    "src/graph_agent.py",
    "src/vector_store.py"
]

all_passed = True
for f in files_to_check:
    full_path = os.path.join(project_root, f)
    if not check_syntax(full_path):
        all_passed = False

if all_passed:
    print("\n🎉 All core files passed syntax check.")
else:
    print("\n⚠️ Some files have syntax errors.")
