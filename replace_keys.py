import os

target_dir = r"e:\jnu\Final_thesis\实验\Raidar\RaidarLLMDetect"
old_key = "YOUR_API_KEY_HERE"
new_key = "YOUR_API_KEY_HERE"

for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if old_key in content:
                    content = content.replace(old_key, new_key)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Replaced key in {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")
