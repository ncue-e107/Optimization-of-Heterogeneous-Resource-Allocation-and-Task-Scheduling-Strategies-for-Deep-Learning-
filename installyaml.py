import yaml
import subprocess

with open('fortheray.yaml','r') as file:
	env_data = yaml.safe_load(file)

pip_packages = env_data['dependencies'][-1]['pip']

for package in pip_packages:
	subprocess.run(['pip','install',package])
	
