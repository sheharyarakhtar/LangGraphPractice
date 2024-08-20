from load_files import DocumentLoaderClass
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

class VectorDBClass(DocumentLoaderClass):
	def __init__(self):
		super().__init__()
		self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4",
            show_progress=False
            )

	def Tokenize(self, chunk_size = 500, chunk_overlap = 50):
		self.text = "\n".join(
			[doc.page_content for doc in self.pages
			])
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size = chunk_size,
			chunk_overlap = chunk_overlap
			)
		self.docs = text_splitter.create_documents([self.text])

	def CreateVectorDB(self, db_type = 'FAISS'):
		if db_type == 'FAISS':
			self.vectorDB = FAISS.from_documents(
				self.docs, 
				self.embedding_model
				)
			self.vectorDB.save_local('faiss_vectorDB')
		if db_type == 'Chroma':
			self.vectorDB = Chroma.from_documents(
				self.docs, 
				self.embedding_model,
				persist_directory = "chroma_db"
				)
	def LoadVectorDB(self, db_type = 'FAISS'):
		if db_type == 'FAISS':
			print("Loading Old FAISS VectorDB")
			self.vectorDB = FAISS.load_local(
				"faiss_vectorDB",
				self.embedding_model
				)
		if db_type == 'Chroma':
			print("Loading Old Chroma VectorDB")
			self.vectorDB = Chroma(
				embedding_function = self.embedding_model,
				persist_directory = "chroma_db"
				)

	def run(self, db_type = 'FAISS'):
		with open('previous_files','rb') as f:
			old = pickle.load(f)
		new_files = self.files.copy()

		if set(new_files) - old:
			self.load_all_files()
			self.Tokenize()
			self.CreateVectorDB(db_type = db_type)
		else:
			self.LoadVectorDB(db_type = db_type)

