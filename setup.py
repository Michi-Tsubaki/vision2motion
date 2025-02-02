from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision2motion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=[
        'setuptools',
        'torch',
        'transformers',
        'pillow',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='michi-tsubaki',
    maintainer_email='michi.tsubaki.tech@gmail.com',
    description='ROS2 framework package that connects vision and motion by using LLM.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_explain_paligemma = vision2motion.visual_explain_paligemma:main',
            'visual_explain_paligemma_en = vision2motion.visual_explain_paligemma_en:main'
        ],
    },
)
