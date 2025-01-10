from setuptools import setup, find_packages


setup(name='plutok',
      version='0.0.1',
      description='low-bitrate tokenizer for audio language models.',
      long_description=open('README.md').read().strip(),
      long_description_content_type='text/markdown',
      keywords=[
            'text-to-speech',
            'tts',
            'voice-clone',
            'zero-shot-tts'
      ],
      author='Zengyi Qin',
      license='MIT License',
      packages=find_packages(),

      python_requires='>=3.9',
      install_requires=[
            'librosa==0.9.1',
            'pydub==0.25.1',
            'numpy==1.22.0',
            'eng_to_ipa==0.0.2',
            'inflect==7.0.0',
            'unidecode==1.3.7',
            'pypinyin==0.50.0',
            'cn2an==0.5.22',
            'jieba==0.42.1',
            'gradio==3.48.0',
            'langid==1.1.6'
      ],
      zip_safe=False
      )
