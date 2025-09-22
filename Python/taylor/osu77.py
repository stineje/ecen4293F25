from pathlib import Path

path = Path("ecommerce/__init__.py")
print(path.exists())
print(path.is_file())
print(path.is_dir())
print(path.name)
print(path.stem)
print(path.suffix)
print(path.parent)
path2 = path.with_name("file.txt")  # create new path obj
print(path2)
print(path2.absolute())
path = path.with_suffix(".txt")  # create new path obj
print(path)
