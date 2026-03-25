import os
import re
import json

posts_dir = "_posts"
projects = {}

pattern = re.compile(r"- \[( |x)\] (.+?) #([a-zA-Z0-9\-]+)")

for filename in os.listdir(posts_dir):
    if not filename.endswith(".md"):
        continue

    with open(os.path.join(posts_dir, filename), "r", encoding="utf-8") as f:
        content = f.read()

        matches = pattern.findall(content)

        for status, text, project in matches:
            if project not in projects:
                projects[project] = {
                    "name": project.replace("-", " ").title(),
                    "done": 0,
                    "total": 0,
                    "todos": []
                }

            projects[project]["total"] += 1

            if status == "x":
                projects[project]["done"] += 1
            else:
                projects[project]["todos"].append(text)

# 生成最终结构
output = {"projects": []}

for p in projects.values():
    progress = int((p["done"] / p["total"]) * 100) if p["total"] > 0 else 0

    output["projects"].append({
        "name": p["name"],
        "progress": progress,
        "todos": p["todos"]
    })

# 写入 JSON
os.makedirs("data", exist_ok=True)

with open("data/projects.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)