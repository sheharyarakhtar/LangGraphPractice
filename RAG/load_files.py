# pip install pypdf
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os

class DocumentLoaderClass():
	def __init__(self):
		self.files = files = os.listdir('./files/')

		self.pdfs = [i for i in files if '.pdf' in i]
		self.docx = [i for i in files if '.docx' in i]
		self.csv = [i for i in files if '.csv' in i]
		self.pages = []	
	def update_loaded_files(self, path):
		with open('loaded_files.txt','a') as f:
			f.write(f"\n {path}")
	def load_pdfs(self):
		for pdf_file in self.pdfs:
			try:
				loader = PyPDFLoader('./files/'+pdf_file)
				pages = loader.load_and_split()
				self.pages.extend(pages)
				self.update_loaded_files(pdf_file)
				print(f"File {pdf_file} loaded")
			except Exception as e:
				print(e)
				print(f"File {pdf_file} not loaded")

	def load_docs(self):
		for docx_file in self.docx:
			try:
				loader = Docx2txtLoader(file_path = './files/'+docx_file)
				pages = loader.load_and_split()
				self.pages.extend(pages)
				self.update_loaded_files(docx_file)
				print(f"File {docx_file} loaded")
			except Exception as e:
				print(e)
				print(f"File {docx_file} not loaded")
	def load_csvs(self):
		for csv_file in self.csv:
			try:
				loader = CSVLoader('./files/'+csv_file)
				pages = loader.load_and_split()
				self.pages.extend(pages)
				self.update_loaded_files(csv_file)
				print(f"File {csv_file} loaded")
			except Exception as e:
				print(e)
				print(f"File {csv_file} not loaded")
	def load_all_files(self):
		if self.pdfs:
			self.load_pdfs()
		if self.docx:
			self.load_docs()
		if self.csv:
			self.load_csvs()

