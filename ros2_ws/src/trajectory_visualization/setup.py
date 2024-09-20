from setuptools import find_packages, setup

package_name = 'trajectory_visualization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['description/astrobee_reference.urdf']),
        ('share/' + package_name, ['description/astrobee_baseline.urdf']),
        ('share/' + package_name, ['description/astrobee_symmetry.urdf']),
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
            'visualize = trajectory_visualization.visualize:main'
        ],
    },
)
