import argparse
from src.ast_parser import parse_file

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="src/file_walker.py")
args = parser.parse_args()

with open(args.file) as f:
    content = f.read()

chunks = parse_file(content, args.file)

print(f"file   : {args.file}")
print(f"chunks : {len(chunks)}")
print("=" * 60)

for i, c in enumerate(chunks):
    name = f"{c.parent_class}.{c.name}" if c.parent_class else c.name
    print(f"[{i}] {c.chunk_type}  {name}  lines {c.start_line}-{c.end_line}")
    print("---")
    print(c.source)
    print("=" * 60)
