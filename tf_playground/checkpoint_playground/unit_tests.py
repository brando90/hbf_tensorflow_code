

'''
units tests to write.

- check that if ./tmp_all_ckpts/ doesn't exist, then it creates the folder but works ()
- check that if nothing inside ./tmp_all_ckpts/ exists experiment runs ()
- check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/ exists but has no hp folders, that it starts doing the
  hp runs from the very first one i.e. hp_stid_1 ()
- check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/ exists, then
    - check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/ has no tf ckpts, in this case
    then start training this hp_N from the first iteration
    - check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/ has tf ckpts, in this case
    continue from this hp AND this tf ckpt (i.e. iteration)

'''
