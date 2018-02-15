from distutils.core import setup

setup(
    name='om_bench',
    version='0.0.1',
    description="Benchmarking tool for generating scaling data and plots.",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='openmdao benchmarking parallel scaling',
    author='OpenMDAO Team',
    author_email='openmdao@openmdao.org',
    license='Apache License, Version 2.0',
    packages=[
        'om_bench',
    ],
    package_data={
        'om_bench.data': ['*.dat'],
    },
)
