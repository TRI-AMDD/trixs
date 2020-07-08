from setuptools import setup, find_packages

# DESCRIPTION must be one line
DESCRIPTION = "TRIXS is a python package enabling analysis and machine learning for X-ray spectroscopy."
LONG_DESCRIPTION = """
TRIXS is a suite of tools to enable analysis, comparison, and machine learning
on X-ray spectroscopy measurements, developed at the 
[Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/).
Current tools focus on X-ray absorption spectroscopy, and are part of a project in
collaboration with JCAP and LBNL.
"""

setup(name="trixs",
      url="https://github.com/TRI-AMDD/trixs",
      version="2020.7.7",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=["scikit-learn==0.23.1",
                        "pymatgen",
                        "imbalanced-learn",
                        "tqdm",
                        "matplotlib",
                        "scipy",
                        "numpy",
                        "monty"
                        ],
      extras_require={
          "tests": ["pytest",
                    "pytest-cov",
                    "coveralls",
                    "memory_profiler"]
      },
      entry_points={
          "console_scripts": [
          ]
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      include_package_data=True,
      author="AMDD - Toyota Research Institute",
      author_email="linda.hung@tri.global",
      maintainer="Linda Hung",
      maintainer_email="linda.hung@tri.global",
      license="Apache",
      keywords=[
          "materials", "chemistry", "science",
          "x-ray", "spectroscopy", "xanes",
          "machine learning", "AI", "artificial intelligence"
      ],
      )