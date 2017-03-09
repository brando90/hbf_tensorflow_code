from multiprocessing import Process
import time

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
    #time.sleep(5)
    print('Done')


##


'''
def train_mdl(args):
    train(mdl,args)

if __name__ == '__main__':
    for mdl_id in range(100):
        # train one model with some specific hyperparms (assume they are chosen randomly inside the funciton bellow or read from a config file or they could just be passed or something)
        p = Process(target=train_mdl, args=(args,))
        p.start()
        p.join()
    print('Done training all models!')
'''
