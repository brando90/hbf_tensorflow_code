

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

- check if ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/mdl_ckpt non exist,
  if no mdl_ckpt (for this hp) then start training this hp_N from the first iteration (WORKS)
- check from what iteration to start ./tmp_all_ckpts/exp_task_name/mdl_nn10/hp_stid_N/mdl_ckpt
  mdl_ckpt, then continue training forom this iteration (for current hp). (TODO)

 - load any checkpoint from the ckpt directory structure (TODO)

'''
