from setuptools import setup, find_packages
from pipreqs.pipreqs import parse_requirements
from codecs import open
import os

HERE = os.path.dirname(__file__)


def read(fname):
    return open(os.path.join(HERE, fname)).read()


install_reqs = parse_requirements(os.path.join(HERE, "requirements.txt"))
install_requires = [str(ir.req) for ir in install_reqs]

setup(
    name="telegramvisualizer",
    version="0.0.1",
    description="Telegram Visualizer allows you to create cool and interesting stats and graphs from Telegram chats",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/somso2e/TelegramVisualizer",
    author="somso2e",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="telegram messanger stats chat tele t.me",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "telegramvis=run:main"
        ],
    }
)
