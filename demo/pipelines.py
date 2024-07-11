from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from .loader import process_document, process_documents, convert_to_base64, plt_img_base64
from PIL import Image
from typing import List
from abc import ABC, abstractmethod
import os

class QAPipeline(ABC):
    def __init__(self, llm="mistral", 
                 img_caption="bakllava", 
                 llm_temp=0.0, 
                 multi_modal_temp=0.0, 
                 oembed_temp=0.0,
                 num_predict=-1,
                 num_thread=None,
                 num_ctx=2048,
                 system=None,
                 template=None,
                 top_k=None,
                 top_p=None,
                 mirostat_tau=None,
                 persist_directory=None):
        self.llm = Ollama(base_url='http://localhost:11434', 
                          model=llm, 
                          temperature=llm_temp, 
                          num_predict=num_predict, 
                          num_thread=num_thread,
                          num_ctx=num_ctx,
                          system=system,
                          template=template,
                          top_k=top_k,
                          top_p=top_p)
        self.multi_modal = Ollama(base_url='http://localhost:11434', 
                                  model=img_caption, 
                                  temperature=multi_modal_temp,
                                  num_predict=num_predict, 
                                  num_thread=num_thread,
                                  num_ctx=num_ctx,
                                  system=system,
                                  template=template,
                                  top_k=top_k,
                                  top_p=top_p)
        self.oembed = OllamaEmbeddings(base_url="http://localhost:11434", 
                                       model=llm, 
                                       temperature=oembed_temp,
                                       mirostat_tau=mirostat_tau,
                                       num_ctx=num_ctx,
                                       num_thread=num_thread,
                                       show_progress=True,
                                       top_k=top_k,
                                       top_p=top_p)
        self.persist_directory = None
        if persist_directory is not None:
            self.vectorstore = Chroma(persist_directory=persist_directory, 
                                      embedding_function=self.oembed)
            self.persist_directory = persist_directory
        else:
            self.vectorstore = None
            self.retriever = None
        self.qachain = None

    @abstractmethod
    def run(self, question: str):
        raise NotImplementedError
    
    def prompt(self, prompt: str):
        return self.llm.predict(prompt)
    
    def run_multi_modal(self, images: List[str], prompt: str, display_img = False):
        images_b64 = []
        for img in images:
            pil_image = Image.open(img)
            file_extension = os.path.splitext(img)[1].lower()
            image_b64 = None
            if "jpg" in file_extension or "jpeg" in file_extension:
                image_b64 = convert_to_base64(pil_image, "JPEG")
                images_b64.append(image_b64)
            elif "png" in file_extension:
                image_b64 = convert_to_base64(pil_image, "PNG")
                images_b64.append(image_b64)
            if display_img:
                plt_img_base64(image_b64)
        llm_with_image_context = self.multi_modal.bind(images=images_b64)
        return llm_with_image_context.invoke(prompt)

class OllamaQAPipeline(QAPipeline): 
    def load_document(self, file: str, parsing_instructions: str, chunk_size = 512, chunk_overlap = 0):
        if self.persist_directory is not None:
            self.vectorstore.add_documents(process_document(file, chunk_size, chunk_overlap, parsing_instructions))
        else:
            self.vectorstore = Chroma.from_documents(documents=process_document(file, chunk_size, chunk_overlap, parsing_instructions), 
                                                     embedding=self.oembed)
        self.qachain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever())
    
    def run(self, question: str):
        return self.qachain({"query": question})['result']

class OllamaMultiQAPipeline(QAPipeline):
    def load_documents(self, source_path: str, parsing_instructions: str, chunk_size = 512, chunk_overlap = 0):
        if self.persist_directory is not None:
            self.vectorstore.add_documents(process_documents(source_path, chunk_size, chunk_overlap, parsing_instructions))
        else:
            self.vectorstore = Chroma.from_documents(documents=process_documents(source_path, chunk_size, chunk_overlap, parsing_instructions), 
                                                     embedding=self.oembed)
        self.qachain = RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever())
    
    def extract_image_data(self, image: str, prompt: str, num_searches=4):
        results = []
        for _ in range(num_searches):
            res = self.run_multi_modal([image], prompt)
            results.append(res)

        for res in results:
            docs = self.vectorstore.similarity_search(res)
            print("----------Prompt Output----------")
            print()
            print(res)
            print()
            print("----------Similarity Search Output----------")
            print()
            for doc in docs:
                print("File: " + doc.metadata["source"])
                print(doc.page_content)
                print()

    def extract_data(self, prompt: str):
        docs = self.vectorstore.similarity_search(prompt)
        print("----------Similarity Search Output----------")
        print()
        for doc in docs:
            print("File: " + doc.metadata["source"])
            print(doc.page_content)
            print()

    def run(self, question: str):
        return self.qachain({"query": question})['result']
