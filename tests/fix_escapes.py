file_path = "d:\\My Professional Projects\\PythonAgent\\learning_lab.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace escaped triple quotes with standard triple quotes
new_content = content.replace('\\"\\"\\"', '"""')

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Fixed escaped quotes in {file_path}")
