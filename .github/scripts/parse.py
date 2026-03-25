import os
import json
import re

POSTS_DIR = "_posts"
OUTPUT_FILE = "data/projects.json"

projects = {}

for filename in os.listdir(POSTS_DIR):
    if not filename.endswith(".md"):
        continue

    with open(os.path.join(POSTS_DIR, filename), "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        match = re.match(r"- \[( |x)\] (.+?) #([\w-]+)", line.strip())
        if match:
            status, text, project = match.groups()

            if project not in projects:
                projects[project] = {
                    "name": project.replace("-", " ").title(),
                    "done": 0,
                    "total": 0,
                    "todos": [],
                    "tasks": []   # ⭐ 新增
                }

            projects[project]["total"] += 1

            task = {
                "text": text,
                "done": status == "x"
            }

            projects[project]["tasks"].append(task)

            if status == "x":
                projects[project]["done"] += 1
            else:
                projects[project]["todos"].append(text)

# 输出
output = {"projects": []}

for p in projects.values():
    progress = int((p["done"] / p["total"]) * 100) if p["total"] > 0 else 0

    output["projects"].append({
        "name": p["name"],
        "progress": progress,
        "todos": p["todos"],
        "tasks": p["tasks"]   # ⭐ 输出 tasks
    })

os.makedirs("data", exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)