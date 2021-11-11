from setuptools import setup, find_packages





setup(name='analysis_of_textual_job_descriptions_with_Pole_Emploi',
      version='0.0.1',
      description='',
      url='https://github.com/adriens63/analysis_of_textual_job_descriptions_with_Pole_Emploi',
      author='Adrien Servière, Efflam Fouques Duparc, Martin Bordes, Victor Michel',
      author_email='adrien.serviere@ensae.fr, efflam.fouquesduparc@ensae.fr, martin.bordes@ensae.fr, victor.michel@ensae.fr',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      install_requires=['numpy', 'tqdm'])