from load_files import DocumentLoaderClass
from langchain.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDBClass(DocumentLoaderClass):
	def __init__(self):
		super().__init__()
		self.text = ""
		self.embeding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4",
            show_progress=False
            )

	def Tokenize(self, chunk_size, chunk_overlap):
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
			self.vectorDB = FAISS.from_documents(self.docs, self.embeding_model)
		if db_type == 'Chroma':
			self.vectorDB = Chroma.from_documents(self.docs, self.embeding_model)
