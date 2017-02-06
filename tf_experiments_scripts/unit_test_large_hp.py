'''
units tests to write.

- check that if ./tmp_all_ckpts doesn't exist, then it creates the folder
  and trains correctly all ckpts (WORKS)
- check that if ./tmp_all_ckpts/exp_task_name doesn't exist, then it creates the folder
  and trains correctly all ckpts (WORKS)
- check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/ doesn't exist, then it creates the
  folder and trains correctly all ckpts (WORKS)

- check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N non exist,
  (i.e. check if all hp_stid_N are missing), if non exist then start doing the
  hp runs from the very first one i.e. hp_stid_1 (WORKS)

- check if ./tmp_all_ckpts/ezxp_task_name/mdl_nn10/hp_stid_N/mdl_ckpt non exist,
  if no mdl_ckpt (for this hp) then start training this hp_N from the first iteration (WORKS)
- check from what iteration to start ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/mdl_ckpt
  mdl_ckpt, then continue training forom this iteration (for current hp). (WORKS)

- load any checkpoint from the ckpt directory structure (WORKS)

- make sure every if statement in the main_large_hp_ckpt is executed (TODO)
  in particular it seems that the one that had arg.restore = True was never ran.
  i.e. the one with the line of code: if does_hp_have_tf_ckpt(path_to_hp_folder): # is there a tf ckpt for this hp?

####

- make sure that a double header is NOT written with the csv writer (TODO)
  i.e. if a specific hp is killed and then re-ran, that it doesn't write again
  the header for the csv file.

- make sure the correct hp params are written to the JSON file (TODO)
  in particular, recall that

- make sure that the correc hps are used upon restoration (TODO)
  in particular notice that when slurm re-runs my config batch script, it will
  re-generate hp's. However, if a tp ckpt file is written, then it should load
  the old hps from the JSON file, not use the newly created ones.

'''
