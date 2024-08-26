from load_files import DocumentLoaderClass
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os
class VectorDBClass(DocumentLoaderClass):
    def __init__(self):
        """
        Initializes the VectorDBClass, inheriting from DocumentLoaderClass.
        Sets up the embedding model using the HuggingFaceEmbeddings with a specific model.
        """
        super().__init__()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4",
            show_progress=False
            )

    def Tokenize(self, chunk_size, chunk_overlap):
        """
        Tokenizes the text content from all loaded pages into smaller chunks.
        
        Args:
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The amount of overlap between consecutive text chunks.
        
        This method concatenates the content of all pages into a single text string, 
        adds source references for each document, and splits the text into smaller chunks.
        The resulting chunks are stored as documents in the 'self.docs' variable.
        """
        self.text = "\n".join(
            [doc.page_content + f"[Source: {doc.metadata['source']}]"
             for doc in self.pages
            ])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
            )
        self.docs = text_splitter.create_documents([self.text])

    def CreateVectorDB(self, db_type='FAISS'):
        """
        Creates a vector database from the tokenized documents.
        
        Args:
            db_type (str): The type of vector database to create ('FAISS' or 'Chroma').
        
        Depending on the 'db_type' argument, this method either creates a FAISS 
        or Chroma vector database from the tokenized documents, using the embedding model. 
        The created database is then saved locally for future use.
        """
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
                persist_directory="chroma_db"
                )

    def LoadVectorDB(self, db_type='FAISS'):
        """
        Loads an existing vector database from local storage.
        
        Args:
            db_type (str): The type of vector database to load ('FAISS' or 'Chroma').
        
        Depending on the 'db_type' argument, this method loads either a FAISS 
        or Chroma vector database from the local storage. 
        The loaded database is assigned to 'self.vectorDB'.
        """
        if db_type == 'FAISS':
            print("Loading Old FAISS VectorDB")
            self.vectorDB = FAISS.load_local(
                "faiss_vectorDB",
                self.embedding_model,
                allow_dangerous_deserialization=True
                )
        
        if db_type == 'Chroma':
            print("Loading Old Chroma VectorDB")
            self.vectorDB = Chroma(
                embedding_function=self.embedding_model,
                persist_directory="chroma_db"
                )

    def run(self, db_type='FAISS', tokenize=False, create_new_db=True, chunk_size=500, chunk_overlap=50):
        """
        Runs the process of either creating or loading a vector database.
        
        Args:
            db_type (str): The type of vector database to use ('FAISS' or 'Chroma').
            tokenize (bool): Whether to tokenize the documents before creating the database.
            create_new_db (bool): Whether to create a new vector database or load an existing one.
            chunk_size (int): The size of each text chunk for tokenization.
            chunk_overlap (int): The amount of overlap between consecutive text chunks.
        
        This method checks for new files and decides whether to create a new vector database 
        or load an existing one. If new files are detected or 'create_new_db' is True, 
        the method tokenizes the documents (if 'tokenize' is True) and creates a new vector database. 
        Otherwise, it loads an existing vector database.
        """
        if os.path.exists('previous_files'):
            with open('previous_files', 'rb') as f:
                old = pickle.load(f)
        else:
            old = set()

        new_files = set(os.listdir('./files/'))
        diff_files = new_files - old
        print("NEW FILES SET", new_files, "OLD FILES SET", old)
        print("DIFF : ", diff_files)

        if diff_files or create_new_db:
            self.load_all_files()
            if tokenize:
                print("Tokenizing All files")
                self.Tokenize(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                print("Keeping original file split")
                self.docs = self.pages
            self.CreateVectorDB(db_type=db_type)
            with open('previous_files', 'wb') as f:
                pickle.dump(new_files, f)
        else:
            self.LoadVectorDB(db_type=db_type)


