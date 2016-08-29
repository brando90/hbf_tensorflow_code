from setuptools import setup

setup(
    name='my_tf_proj', #project name
    version='0.1.0',
    description='my research library for deep learning',
    #url
    author='Brando Miranda',
    author_email='brando90@mit.edu',
    license='MIT',
    packages=['my_tf_pkg'],
    #install_requires=['numpy','keras','namespaces','pdb','scikit-learn','scipy','virtualenv']
    install_requires=['numpy','keras','namespaces==1.0.0','pdb','virtualenv']
)
