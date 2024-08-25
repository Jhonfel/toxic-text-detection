from setuptools import setup, find_packages

print(find_packages(where="src"))

setup(
    name="toxic_text_detection",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
