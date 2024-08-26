# pip install pypdf
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import os
import pickle

class DocumentLoaderClass():
    def __init__(self):
        """
        Initializes the DocumentLoaderClass.
        
        This method lists all the files in the './files/' directory and categorizes them
        into PDF, DOCX, and CSV files. It also initializes an empty list to store the loaded pages.
        """
        self.files = os.listdir('./files/')
        
        # Categorize files based on their extensions.
        self.pdfs = [i for i in self.files if '.pdf' in i]
        self.docx = [i for i in self.files if '.docx' in i]
        self.csv = [i for i in self.files if '.csv' in i]
        
        # Initialize an empty list to hold the content of loaded documents.
        self.pages = []    
    
    def update_loaded_files(self, path):
        """
        Appends the path of a successfully loaded file to 'loaded_files.txt'.
        
        Args:
            path (str): The file path to be recorded.
        
        This method is used to keep track of files that have been successfully loaded.
        """
        with open('loaded_files.txt', 'a') as f:
            f.write(f"\n {path}")
    
    def load_pdfs(self):
        """
        Loads and splits the content of all PDF files in the './files/' directory.
        
        This method uses PyPDFLoader to load and split the content of each PDF file
        into individual pages, which are then added to the 'self.pages' list.
        The method also records the loaded files and handles exceptions if a file fails to load.
        
        Returns:
            str: The path of the last loaded PDF file.
        """
        for pdf_file in self.pdfs:
            try:
                loader = PyPDFLoader('./files/' + pdf_file)
                pages = loader.load_and_split()
                self.pages.extend(pages)
                self.update_loaded_files(pdf_file)
                print(f"File {pdf_file} loaded")
            except Exception as e:
                print(e)
                print(f"File {pdf_file} not loaded")
        return './files/' + pdf_file

    def load_docs(self):
        """
        Loads and splits the content of all DOCX files in the './files/' directory.
        
        This method uses Docx2txtLoader to load and split the content of each DOCX file
        into individual pages, which are then added to the 'self.pages' list.
        The method also records the loaded files and handles exceptions if a file fails to load.
        
        Returns:
            str: The path of the last loaded DOCX file.
        """
        for docx_file in self.docx:
            try:
                loader = Docx2txtLoader(file_path='./files/' + docx_file)
                pages = loader.load_and_split()
                self.pages.extend(pages)
                self.update_loaded_files(docx_file)
                print(f"File {docx_file} loaded")
            except Exception as e:
                print(e)
                print(f"File {docx_file} not loaded")
        return './files/' + docx_file
    
    def load_csvs(self):
        """
        Loads and splits the content of all CSV files in the './files/' directory.
        
        This method uses CSVLoader to load and split the content of each CSV file
        into individual pages, which are then added to the 'self.pages' list.
        The method also records the loaded files and handles exceptions if a file fails to load.
        
        Returns:
            str: The path of the last loaded CSV file.
        """
        for csv_file in self.csv:
            try:
                loader = CSVLoader('./files/' + csv_file)
                pages = loader.load_and_split()
                self.pages.extend(pages)
                self.update_loaded_files(csv_file)
                print(f"File {csv_file} loaded")
            except Exception as e:
                print(e)
                print(f"File {csv_file} not loaded")
        return './files/' + csv_file
    
    def load_all_files(self):
        """
        Loads and splits the content of all files (PDF, DOCX, CSV) in the './files/' directory.
        
        This method sequentially loads all PDF, DOCX, and CSV files, adding the content
        of each file to the 'self.pages' list. It also tracks the loaded files and updates
        the 'previous_files' record with the paths of the newly loaded files.
        """
        files = []
        if self.pdfs:
            files.append(self.load_pdfs())
        if self.docx:
            files.append(self.load_docs())
        if self.csv:
            files.append(self.load_csvs())
        
        # Convert the list of loaded files to a set.
        self.files = set(files)
        
        # Save the record of previously loaded files, if not already existing.
        filename = 'previous_files'
        if not os.path.exists(filename):
            with open(filename, 'wb') as file:
                pickle.dump(self.files, file)
        else: 
            pass

		

