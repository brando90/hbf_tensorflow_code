import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num", help="help text", type=int)
parser.add_argument("-f", "--foo", help="foo help text")
parser.add_argument("-b", "--bar", help="bar help text", action='store_true')
args = parser.parse_args()

print( args.bar )
print( args.foo )
print( args.num )
