# Load the dictionary back from the pickle file.
import pickle
import namespaces as ns

# arg_dict = dict(arg).copy()
# arg_dict = get_remove_functions_from_dict(arg_dict)
# pickle.dump( arg_dict, open( "slurm-%s_%s.p"%(arg.slurm_jobid,arg.slurm_array_task_id) , "wb" ) )

slurm_jobid,slurm_array_task_id = 1,2
arg = pickle.load( open( "slurm-%s_%s.p"%(slurm_jobid,slurm_array_task_id), "rb" ) )
print(arg)

print('ns_args: ', ns.Namespace(arg) )
