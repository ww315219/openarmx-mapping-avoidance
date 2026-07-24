from glob import glob
from setuptools import find_packages, setup


package_name = "openarmx_obstacle_avoidance"


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
    description="ESDF-based local obstacle avoidance filters for OpenArmX teleoperation.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bimanual_esdf_avoidance_filter = openarmx_obstacle_avoidance.bimanual_esdf_avoidance_filter:main",
            "bimanual_modal_observer = openarmx_obstacle_avoidance.bimanual_modal_observer:main",
            "bimanual_esdf_predictive_planner = openarmx_obstacle_avoidance.bimanual_esdf_predictive_planner:main",
            "bimanual_robot_esdf_clearer = openarmx_obstacle_avoidance.bimanual_robot_esdf_clearer:main",
            "safety_metrics_recorder = openarmx_obstacle_avoidance.safety_metrics_recorder:main",
            "right_arm_cumotion_avoidance_filter = openarmx_obstacle_avoidance.right_arm_cumotion_avoidance_filter:main",
            "right_arm_esdf_avoidance_filter = openarmx_obstacle_avoidance.right_arm_esdf_avoidance_filter:main",
            "right_arm_esdf_predictive_planner = openarmx_obstacle_avoidance.right_arm_esdf_predictive_planner:main",
        ],
    },
)
