# imports the __init__.py module which in this special example has loaded all the modules in its package/folder
# so you can use all your functions in that package due to the imports in the __ini__.py from pkg_1
import pkg_1
# line bellow does not help to run f2. Notice that pip (package manager) points to
# my_proj directiory. This probably means it needs the package names to import things since sys.path points to
# my_project. So say the project name to help sys.path find your modules (that should be inside packages)
# import module1 as m1

pkg_1.f1()
