from setuptools import setup, find_packages

# PyPI / pip require a normalized name (no spaces). Display name can stay in readme / UI.
setup(
    name="frp-chatbot",
    version="0.0.1",
    author="Usama Ali",
    author_email="usamauet28@gmail.com",
    description="Paper based chatbot for FRP",
    packages=find_packages(),
    install_requires=[],
)

# find_packages() will look for all the packages in the src directory.
# it will look for __init__.py and consider it as local packages.
