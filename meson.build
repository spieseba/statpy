project('statpy', version:'0.1')

pymod = import('python')
python = pymod.find_installation('python3')
python.get_extra_kwargs()['cmdclass']['git_commit_hash'] = git_hash
pydep = python.dependency()
subdir('src/statpy/')