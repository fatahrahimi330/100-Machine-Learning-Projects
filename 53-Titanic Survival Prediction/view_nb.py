import json

with open("TitanicSurvivalPrediction.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    source = "".join(cell.get("source", []))
    print(source)

