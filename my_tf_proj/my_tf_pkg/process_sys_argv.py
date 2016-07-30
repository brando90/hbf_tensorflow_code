import pdb

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError('Cannot conver %s to bool'%s)

def process_argv(argv):
    print 'print argv =',argv
    print 'len(argv) =',len(argv)
    experiment_name = 'tmp_experiment'
    train_S_type = 'multiple_S'
    units_list = [24,24]
    # units_list = [96,96]
    # task_name = 'task_qianli_func'
    # task_name = 'task_hrushikesh'
    # re_train = None
    # task_name = 'task_f_2D_task2'
    task_name = 'task_f_2d_task2_xsinglog1_x_depth2'
    experiment_root_dir = 'om_xsinlog1_x_depth2'
    # task_name = 'task_f_2d_task2_xsinglog1_x_depth3'
    # task_name = 'task_MNIST_flat'
    #
    #bn = True
    #bn = False
    print '---------> len(argv)', len(argv)
    if is_it_tensorboard_run(argv):
        if len(argv) == 7:
            # python main_nn.py slurm_jobid slurm_array_task_id job_name mdl_save bn --logdir=/tmp/mdl_logs
            # python main_nn.py slurm_jobid slurm_array_task_id job_name_TEST True False --logdir=/tmp/mdl_logs
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            job_name = argv[3]
            mdl_save = bool(argv[4])
            bn = argv[5]
            print 1
        else:
            # python main_nn.py --logdir=/tmp/mdl_logs
            slurm_jobid = 'TB'
            slurm_array_task_id = 'TB'
            job_name = 'TB'
            mdl_save = True
            print 2
    else:
        mdl_save = True
        if len(argv) == 11:
            # python main_nn.py slurm_jobid slurm_array_task_id experiment_root_dir experiment_name job_name mdl_save 3,3 multiple_S/single_S task_name bn

            # python main_nn.py slurm_jobid slurm_array_task_id om_xsinlog1_x_depth2 experiment_name job_name True 3,3 multiple_S task_f_2d_task2_xsinglog1_x_depth2 False
            slurm_jobid = argv[1]
            slurm_array_task_id = argv[2]
            experiment_root_dir = argv[3]
            job_name = argv[4]
            experiment_name = argv[5]
            mdl_save = bool(argv[6])
            units =  argv[7].split(',')
            units_list = [ int(a) for a in units ]
            train_S_type = argv[8] # multiple_S/single_S
            task_name = argv[9]
            bn = argv[10]
            print 2.8
        # elif len(argv) == 9:
        #     # python main_nn.py      slurm_jobid     slurm_array_task_id     job_name      True            experiment_name 3,3,3  multiple_S/single_S
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True  experiment_name 3 multiple_S/single_S
        #     #
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True  experiment_name 2,2 multiple_S
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True  experiment_name 2,2 single_S
        #     slurm_jobid = argv[1]
        #     slurm_array_task_id = argv[2]
        #     job_name = argv[3]
        #     mdl_save = bool(argv[4])
        #      = argv[5]
        #     experiment_name = argv[6]
        #     units =  argv[7].split(',')
        #     units_list = [ int(a) for a in units ]
        #     train_S_type = argv[8] # multiple_S/single_S
        #     print 2.8
        # elif len(argv) == 8:
        #     # python main_nn.py      slurm_jobid     slurm_array_task_id     job_name      True            experiment_name 3,3,3
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True  experiment_name 3
        #     #
        #     slurm_jobid = argv[1]
        #     slurm_array_task_id = argv[2]
        #     job_name = argv[3]
        #     mdl_save = bool(argv[4])
        #      = argv[5]
        #     experiment_name = argv[6]
        #     units =  argv[7].split(',')
        #     units_list = [ int(a) for a in units ]
        #     print 2.9
        # elif len(argv) == 7:
        #     # python main_nn.py      slurm_jobid     slurm_array_task_id     job_name      True            experiment_name
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True  experiment_name
        #     #
        #     slurm_jobid = argv[1]
        #     slurm_array_task_id = argv[2]
        #     job_name = argv[3]
        #     mdl_save = bool(argv[4])
        #      = argv[5]
        #     experiment_name = argv[6]
        #     print 3
        # elif len(argv) == 6:
        #     # python main_nn.py      slurm_jobid     slurm_array_task_id     job_name      True
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True
        #     slurm_jobid = argv[1]
        #     slurm_array_task_id = argv[2]
        #     job_name = argv[3]
        #     mdl_save = bool(argv[4])
        #      = argv[5]
        #     print 4
        # elif len(argv) == 5:
        #     # python main_nn.py      slurm_jobid     slurm_array_task_id     job_name      True
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name True
        #      = 'tmp_om'
        #     slurm_jobid = argv[1]
        #     slurm_array_task_id = argv[2]
        #     job_name = argv[3]
        #     mdl_save = bool(argv[4])
        #     print 5
        # elif len(argv) == 4:
        #     # python main_nn.py slurm_jobid slurm_array_task_id job_name
        #      = 'om'
        #     slurm_jobid = argv[1]
        #     slurm_array_task_id = argv[2]
        #     job_name = argv[3]
        #     mdl_save = True
        #     print 6
        # elif len(argv) == 2: # if job_name
        #     # python main_nn.py job_name
        #     ='tmp'
        #     slurm_jobid = '0'
        #     slurm_array_task_id = '00'
        #     job_name = argv[1]
        #     print 7
        # elif len(argv) == 1:
        #     # python main_nn.py
        #     ='tmp'
        #     slurm_jobid = '0'
        #     slurm_array_task_id = '00'
        #     job_name = 'test'
        #     print 8
        else:
            raise ValueError('Need to specify the correct number of params')
    bn = str_to_bool(bn)
    return (experiment_root_dir,slurm_jobid,slurm_array_task_id,job_name,mdl_save,experiment_name,units_list,train_S_type,task_name,bn)

def is_it_tensorboard_run(argv):
    check_args = []
    for sys_arg in argv:
        check_args.extend(sys_arg.split('='))
    print check_args
    return '--logdir' in check_args
