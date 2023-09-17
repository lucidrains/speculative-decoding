from setuptools import setup, find_packages

setup(
  name = 'speculative-decoding',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Speculative Decoding',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/speculative-decoding',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'efficient decoding'
  ],
  install_requires=[
    'beartype',
    'einops>=0.6.1',
    'rotary-embedding-torch>=0.3.0',
    'torch>=1.12',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
