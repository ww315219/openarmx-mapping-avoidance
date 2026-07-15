from glob import glob

from setuptools import find_packages, setup


package_name = "openarmx_visual_cues"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="wanghua",
    maintainer_email="wanghua@example.com",
    description="RViz visual cues for OpenArmX bimanual teleoperation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bimanual_visual_cues = openarmx_visual_cues.bimanual_visual_cues:main",
        ],
    },
)
