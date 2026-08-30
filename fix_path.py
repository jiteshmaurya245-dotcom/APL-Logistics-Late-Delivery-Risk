path = "app/streamlit_dashboard.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_line = "df = pd.read_csv('APL_Logistics.csv', encoding='latin1')"
new_line = "df = pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / 'APL_Logistics.csv', encoding='latin1')"

if old_line in content:
    content = content.replace(old_line, new_line)
    print("Fixed the read_csv line.")
else:
    print("Old line not found - no change made to read_csv.")

if "from pathlib import Path" not in content:
    content = content.replace("import pandas as pd", "import pandas as pd\nfrom pathlib import Path", 1)
    print("Added Path import.")
else:
    print("Path import already present.")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done.")