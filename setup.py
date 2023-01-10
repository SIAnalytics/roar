from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='roar',
        version='0.0.1',
        description='Toolbox for estimating a feature importance',
        keywords='computer vision, image classification, explainable ai',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        license='Apache License 2.0',
        zip_safe=False)
