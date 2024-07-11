from typing import List
import os
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import spacy
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from tqdm import tqdm
import glob
import base64
from io import BytesIO
from PIL import Image
from IPython.display import HTML, display

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
poppler_path = r'/usr/local/bin'
nlp = spacy.load("en_core_web_sm")

LOADER_MAPPING = {
    ".csv": (CSVLoader, dict()),
    ".doc": (UnstructuredWordDocumentLoader, dict()),
    ".docx": (UnstructuredWordDocumentLoader, dict()),
    ".enex": (EverNoteLoader, dict()),
    ".epub": (UnstructuredEPubLoader, dict()),
    ".html": (UnstructuredHTMLLoader, dict()),
    ".md": (UnstructuredMarkdownLoader, dict()),
    ".odt": (UnstructuredODTLoader, dict()),
    ".pdf": (None, dict()),  
    ".PDF": (None, dict()),
    ".ppt": (UnstructuredPowerPointLoader, dict()),
    ".pptx": (UnstructuredPowerPointLoader, dict()),
    ".txt": (TextLoader, dict({"encoding": "utf8"}))
}

def pdf_to_images(file_path: str) -> List:
    return convert_from_path(file_path, poppler_path=poppler_path)

def ocr_image(image) -> str:
    return pytesseract.image_to_string(image)

def extract_text_with_pdfplumber(file_path: str) -> str:
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_tables(file_path: str) -> List[str]:
    with pdfplumber.open(file_path) as pdf:
        tables = []
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                tables.append(table)
        return tables

def process_text_with_spacy(text: str) -> str:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(f"{ent.label_}: {ent.text}")
    return "\n".join(entities)

def apply_parsing_instructions(text: str, tables: List[str], parsing_instructions: str) -> str:
    parsed_text = text
    for table in tables:
        parsed_text += "\n" + str(table)
    parsed_text += "\n\nExtracted Entities:\n" + process_text_with_spacy(text)
    parsed_text += "\n\nParsing Instructions Used:\n" + parsing_instructions
    return parsed_text

def load_pdf_with_ocr(file_path: str, parsing_instructions: str) -> List[Document]:
    images = pdf_to_images(file_path)
    text = ''
    for image in images:
        text += ocr_image(image)
    tables = extract_tables(file_path)
    text = apply_parsing_instructions(text, tables, parsing_instructions)
    documents = [Document(page_content=text, metadata={"source": file_path})]
    return documents

def load_pdf_with_pdfplumber(file_path: str, parsing_instructions: str) -> List[Document]:
    text = extract_text_with_pdfplumber(file_path)
    tables = extract_tables(file_path)
    text = apply_parsing_instructions(text, tables, parsing_instructions)
    documents = [Document(page_content=text, metadata={"source": file_path})]
    return documents

def load_single_document(file_path: str, parsing_instructions: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        if loader_class is None and (ext == ".pdf" or ext == ".PDF"):
            return load_pdf_with_pdfplumber(file_path, parsing_instructions) + load_pdf_with_ocr(file_path, parsing_instructions)
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" width="320" height="240"/>'
    display(HTML(image_html))

def convert_to_base64(pil_image, type):
    buffered = BytesIO()
    pil_image.save(buffered, format=type)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def process_document(file: str, chunk_size=512, chunk_overlap=0, parsing_instructions: str='') -> List[Document]:
    documents = load_single_document(file, parsing_instructions)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def load_doc(file: str, parsing_instructions: str) -> List[Document]:
    return load_single_document(file, parsing_instructions)

def process_documents(files: str, chunk_size=512, chunk_overlap=0, parsing_instructions: str='') -> List[Document]:
    documents = load_docs(files, parsing_instructions)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def load_docs(files: str, parsing_instructions: str) -> List[Document]:
    all_files = glob.glob(os.path.join(files, "*"))
    results = []
    for file in all_files:
        results.extend(load_single_document(file, parsing_instructions))
    return results
