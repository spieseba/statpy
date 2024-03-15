project('statpy', version:'0.1')

pymod = import('python')
python = pymod.find_installation('python3')
extra_kwargs = pymod.find_installation().get_extra_kwargs()
pydep = python.dependency()
subdir('src/statpy/')