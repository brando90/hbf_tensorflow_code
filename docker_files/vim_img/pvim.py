import sys

from subprocess import call

def make_prefix(pwd_array):
    #print(pwd_array)
    string = ''
    for path_part in pwd_array:
        #print(path_part)
        string = string + '/' + path_part
    string = string + '/'
    return string

def get_prefix_for_docker_env(pwd):
    '''

    e.g. returns something like /home_simulation_research/hbf_tensorflow_code/docker_files/vim_img/
    '''
    pwd_split = pwd.split('/')
    for i in range( len(pwd_split) ):
        path_part = pwd_split[i]
        #print(path_part)
        if path_part == 'home_simulation_research':
            pwd_split_str = make_prefix(pwd_split[i:])
            break
    return pwd_split_str

if len(sys.argv) < 3:
    raise ValueError('Need to provide a file to vim')
pwd, filename = sys.argv[1], sys.argv[2]
pwd_split_str = get_prefix_for_docker_env(pwd)
print(pwd_split_str+filename)
