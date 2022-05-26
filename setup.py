from setuptools import setup

setup(
    name='planerecnet',
    version='0.1.0',
    description='A plane detection deep neural network',
    url='https://github.com/ishrat-tl/PlaneRecNet',
    author='Yaxu Xie',
    author_email='yaxu0325@gmail.com',
    license='MIT License',
    packages=['planerecnet'],
    install_requires=['torch',
                      'torchvision',
                      'torchaudio',
                      'opencv-python',
                      'scipy',
                      'numpy',
                      'tensorboardX'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    info='This is an official implementation for PlaneRecNet: A multi-task convolutional neural network provides '
         'instance segmentation for piece-wise planes and monocular depth estimation, and focus on the cross-task '
         'consistency between two branches '
)
