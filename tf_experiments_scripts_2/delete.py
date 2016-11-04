import os
import json

def get_all_simulation_results(path_to_all_experiments_for_task,verbose=True):
    '''
    Given a path to all the experiments for a specific task, goes through each individual folder for each model
    and returns ALL results a specific model.

    If a model has more than one dirpath, we return the results for *both*.
    Also, for a specific model we might have e.g. 200 runs. Each run has lots of iterations.

    returns { model: [best_over_run_i]^200_i } = { model: [ min runs_j ]^200_i }
    where best_over_run_i = min over all the iterions
    '''
    expts_best_results = {} #maps units -> to corresponding best data (note: keys are numbers so it can't be a namespace)
    for (dirpath, dirnames, filenames) in os.walk(top=path_to_all_experiments_for_task,topdown=True):
        print(filenames)
        #if (dirpath != path_to_all_experiments_for_task) and (not 'mdls' in dirpath): # if current dirpath is a valid experiment and not . (itself)
        for filename in filenames: #for run in all_runs
            if 'json' in filename: # if current run=filenmae is a json struct then it has the results
                with open(dirpath+'/'+filename, 'r') as data_file:
                    results_current_run = json.load(data_file)
                #print(results_current_run)
                nb_units = _get_nb_units(results_current_run)
                #print(results_current_run)
                if nb_units == 20:
                    os.remove(filename)
                    print(filename)
    return expts_best_results

def _get_nb_units(results):
    '''
    gets the number of units for these results
    '''
    return results['arg_dict']['dims'][1] if not 'dims' in results else results['dims'][1]

print('start')
get_all_simulation_results('.',verbose=True)
print('end')
