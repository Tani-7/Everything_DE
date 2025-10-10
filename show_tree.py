from pathlib import Path

def print_tree(root: Path, prefix: str = ""):
    """Recursively print folder tree starting from root."""
    print(prefix + root.name + "/")
    for path in sorted(root.iterdir()):
        if path.is_dir():
            print_tree(path, prefix + "    ")
        else:
            print(prefix + "    " + path.name)

folders_to_show = ["src", "models"]

for folder in folders_to_show:
    path = Path(folder)
    if path.exists():
        print_tree(path)
    else:
        print(f"{folder}/ not found")
