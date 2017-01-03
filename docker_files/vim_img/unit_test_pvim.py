import pvim

pvim.hello_world()

pwd, filename = '/other_stuff/asdasd/asdasd/home_simulation_research/hbf_tensorflow_code/docker_files/vim_img', 'pvim.py'
output = pvim.print_path_plus_filename(pwd, filename)
answer = '/home_simulation_research/hbf_tensorflow_code/docker_files/vim_img/pvim.py'
print( answer == output )
