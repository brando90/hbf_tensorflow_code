from time import sleep
import functools

print = functools.partial(print, flush=True)

var1 = 'var1'
var2 = 'var2'
for i in range(100):
    sleep(3)
    print('i = ', i)
    print(var1)
    print(var2)
