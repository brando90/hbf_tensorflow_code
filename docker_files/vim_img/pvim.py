import sys

from subprocess import call

def make_prefix(pwd_array):
    print(pwd_array)
    string = ''
    for path_part in pwd_array:
        #print(path_part)
        string = string + '/' + path_part
    string = string + '/'
    return string


print(sys.argv)

pwd = sys.argv[1]
filename = sys.argv[2]
# print(pwd)
# print(filename)
#call(["vim", pwd+'/'+filename])

pwd_split = pwd.split('/')
for i in range( len(pwd_split) ):
    path_part = pwd_split[i]
    print(path_part)
    if path_part == 'home_simulation_research':
        pwd_split_str = make_prefix(pwd_split[i:])
        break
print(pwd_split_str+filename)
call(["vim", pwd_split_str+filename])
