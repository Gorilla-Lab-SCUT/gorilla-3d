import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--class_name", type=str, nargs=2, default=["123", "456"])
args = parser.parse_args()

print(args.class_name)
