import functools

#print = functools.partial(print, flush=True)
#print( print )

def f():
    print = functools.partial(print, flush=True)
    print( print )

f()
