import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num", help="help text", type=int)
args = parser.parse_args()

print( args.num )
