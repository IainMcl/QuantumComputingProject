from setuptools import setup

setup(name='qc_simulator',
    version='0.1',
    description='A basic quantum computer simulator',
    url='https://github.com/lululaplap/QuantumComputingProject',
    author='Andreas Malekos',
    autor_email='andymalekos@gmail.com',
    packages=['qc_simulator'],
    install_requires=['matplotlib', 'numpy==1.14', 'scipy', 'qutip', 'abc'],
    zip_false=False)
