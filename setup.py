import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ambf6dpose",
    version="0.0.0",
    author="Juan Antonio Barragan",
    author_email="jbarrag3@jh.edu",
    description="A package for generating 6D pose estimation data using the AMBF simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "rich", "click"],
    include_package_data=True,
    python_requires=">=3.8",
    # entry_points={
    #     "console_scripts": [
    #         "surg_seg_generate_labels = surg_seg.Scripts.generate_labels:main",
    #         "surg_seg_ros_video_record = surg_seg.Scripts.RosVideoRecord.ros_video_record:main",
    #     ]
    # },
)
