# from PKG/MODULE import Function or from PKG/MODULE import Variable or from PKG/MODULE import THINGIE
# e.g from math import pi
from require import require

#require is a function
#use a path (relative or not)
utils = require('../../utils.py') # now everything in utils is available through utils variable

pedro = require('../../utils.py').name
pedro = utils.name

utils.greet(pedro)
