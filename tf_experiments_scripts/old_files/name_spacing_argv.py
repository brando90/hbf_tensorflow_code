import sys

sys.argv[1]
sys.argv[2]

arg_names = ['param1', 'initalize2', 'brando']
dict_of_stuff = dict(zip(arg_names, sys.argv[1:]))
args = ns.FrozenNamespace(dict_of_stuff)

nums = [1,2,3]
letters = [a,b,c]
zip(nums, letters) -> [(1,a), (2,b), (3,c)]
