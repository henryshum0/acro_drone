from setuptools import find_packages, setup

package_name = 'acro_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='henryshum0',
    maintainer_email='henryshum0@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'privileged_agent = acro_drone.nodes.privileged_agent:main',
			'main_node = acro_drone.nodes.main_node:main',
			'return_node = acro_drone.nodes.return_node:main',
        ],
    },
)
