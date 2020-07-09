from setuptools import setup, find_packages

# DESCRIPTION must be one line
DESCRIPTION = "Data analysis and machine learning for X-ray spectroscopy"
LONG_DESCRIPTION = """
[Toyota Research Institute](http://www.tri.global/accelerated-materials-design-and-discovery/)
 X-ray Spectroscopy (TRIXS) is a suite of tools that enables analysis, comparison, and machine learning
for X-ray spectroscopy measurements.
Currently available tools focus on X-ray absorption spectroscopy, particularly XANES spectra.

"""

setup(name="trixs",
      url="https://github.com/TRI-AMDD/trixs",
      version="2020.7.8",
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
                        "numpy"
                        ],
      extras_require={
          "tests": ["pytest",
                    "pytest-cov",
                    "coveralls",
                    "memory_profiler"],
          "dev": ["pytest",
                  "ipywidgets",
                  "jupyterlab",
                  "setuptools",
                  "wheel",
                  "twine"]
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
