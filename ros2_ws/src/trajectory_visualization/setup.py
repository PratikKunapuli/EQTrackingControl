from setuptools import find_packages, setup

package_name = 'trajectory_visualization'


from pathlib import Path

rootdir = "."
for extension in 'urdf'.split():
    for path in Path(rootdir).glob('*.' + extension):
        print("match: " + path)
        
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name, ['description/*.urdf']),
        # ('share/' + package_name, ['description/astrobee_reference.urdf']),
        # ('share/' + package_name, ['description/astrobee_baseline.urdf']),
        # ('share/' + package_name, ['description/astrobee_symmetry.urdf']),
        ('share/' + package_name, [
            'description/astrobee_reference.urdf',
            'description/astrobee_baseline.urdf',
            'description/astrobee_symmetry.urdf',
            'description/quadrotor_reference.urdf',
            'description/quadrotor_baseline.urdf',
            'description/quadrotor_symmetry.urdf'
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jake',
    maintainer_email='jwelde@seas.upenn.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visualize = trajectory_visualization.visualize:main',
            'eval_visualize = trajectory_visualization.eval_visualize:main'
        ],
    },
)
