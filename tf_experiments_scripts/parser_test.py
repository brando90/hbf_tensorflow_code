import argparse

print('parser_test running...')
parser = argparse.ArgumentParser()
#parser.add_argument("-d", "--debug", help="debug mode: loads the old (pickle) config file to run in debug mode", action='store_true')
#parser.add_argument("-tj", "--type_job", help="type_job for run")
parser.add_argument("-sj", "--SLURM_JOBID", help="SLURM_JOBID for run")

cmd_args = parser.parse_args()
print()
