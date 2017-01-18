get_k = lambda a: 7*a + 4*3*2*a**2
shallow = lambda k: 4*k+k+k
bt = lambda f: 2*f+f +2*f*(2*f)+ 2*(2*f)

print( 'shallow(%s), bt(%s)'%(31,6), shallow(31), bt(6) )
print( 'shallow(%s), bt(%s)'%(110,12), shallow(110), bt(12) )
print( 'shallow(%s), bt(%s)'%(237,36), shallow(237), bt(18) )

get_k = lambda a: 13*a + 200*a**2
shallow = lambda k: 8*k+k+k
bt = lambda f: 13*f+20*f**2
