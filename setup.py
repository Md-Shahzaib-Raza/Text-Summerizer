import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Text-Summerizer"
AUTHER_USER_NAME = "Shahzaib"
SRC_REPO = "textSummarizer"
AUTHER_EMAIL = "mdshahzaibraza1234@gmail.com"




setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHER_USER_NAME,
    author_email=AUTHER_EMAIL,
    description="A small python package for NLP app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHER_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHER_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)