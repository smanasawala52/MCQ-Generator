from setuptools import find_packages, setup
setup(
    name='mcqgenerator',
    version='0.0.1',
    author='shabbir manasawala',
    author_email='shabbir.manasawala52@gmail.com',
    install_requires=["openai", "langchain", "langchain_community","streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages()
)