import yaml
import os
from vectorDB import VectorDBClass
with open('config.yaml', 'r') as file:
	config = yaml.safe_load(file)
os.environ['GOOGLE_API_KEY'] = config['GOOGLE_API_KEY']
os.environ['HF_TOKEN'] = config['HF_TOKEN']
os.environ['HF_HOME'] = config['HF_HOME']
loader = VectorDBClass()
loader.run(db_type = 'Chroma')