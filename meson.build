git_sha = vcs_tag(command: ['git', 'rev-parse', 'HEAD'], 
                  output: 'git_sha.txt') 

project('statpy', version: '0.1')

pymod = import('python')
python = pymod.find_installation('python3')
pydep = python.dependency()
subdir('src/statpy/')