import sys

from subprocess import call

def _make_prefix(pwd_array):
    '''
    Given pwd_array concatenates all the elements of the array and  puts a / between them.

    e.g.
    input = [ "home_simulation_research", "hbf_tensorflow_code", "docker_files", "vim_img"]
    output = "/home_simulation_research/hbf_tensorflow_code/docker_files/vim_img/"
    '''
    #print(pwd_array)
    string = ''
    for path_part in pwd_array:
        #print(path_part)
        string = string + '/' + path_part
    string = string + '/'
    return string

def get_prefix_for_docker_env(pwd):
    '''
    Returns the prefix string relative to my  docker filesystem.

    e.g.
    input = '/Users/brandomiranda/home_simulation_research/hbf_tensorflow_code/docker_files/vim_img'
    output = /home_simulation_research/hbf_tensorflow_code/docker_files/vim_img/
    '''
    pwd_split = pwd.split('/')
    for i in range( len(pwd_split) ):
        path_part = pwd_split[i]
        #print(path_part)
        if path_part == 'home_simulation_research':
            pwd_split_str = _make_prefix(pwd_split[i:])
            break
    return pwd_split_str

if len(sys.argv) < 3:
    raise ValueError('Need to provide a file to vim')
pwd, filename = sys.argv[1], sys.argv[2]
pwd_split_str = get_prefix_for_docker_env(pwd)
print(pwd_split_str+filename)
