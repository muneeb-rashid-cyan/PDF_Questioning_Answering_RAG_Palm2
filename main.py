from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (DocxReader,HWPReader,PDFReader,EpubReader,FlatReader,HTMLTagReader,ImageCaptionReader,ImageReader,ImageVisionLLMReader,IPYNBReader,MarkdownReader,MboxReader,PptxReader,PandasCSVReader,VideoAudioReader,UnstructuredReader,PyMuPDFReader,ImageTabularChartReader,XMLReader,PagedCSVReader,CSVReader,RTFReader)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.llms.palm import PaLM
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
# from dotenv import load_dotenv
# load_dotenv()



import nest_asyncio 
nest_asyncio.apply()

# Load PALM api key from .env file
palm_api_key = "AIzaSyA-0q65OugleWkbAy4XWhCEuR1SKvYiORU"

# Configure hugging face embeding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)


# initialize PALM model
model = PaLM(api_key=palm_api_key)



parser = PDFReader()


# file_extractor = {".docx": parser, ".pdf": parser,".epub": parser}

file_extractor = { ".pdf": parser}
# HTML Tag Reader example
documents = SimpleDirectoryReader(
    "data", file_extractor=file_extractor
).load_data()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
nodes = parser.get_nodes_from_documents(documents)

print(nodes[5].text)


# Chunking (converting the document into nodes)
# node_parser = SentenceWindowNodeParser.from_defaults(
#     # how many sentences on either side to capture
#     window_size=6,
#     # the metadata key that holds the window of surrounding sentences
#     window_metadata_key="window",
#     # the metadata key that holds the original sentence
#     original_text_metadata_key="original_sentence",
# )



# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        parser,
        HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5"),
    ]
)


# run the pipeline to extract nodes
nodes = pipeline.run(documents=documents)

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

chroma_client = chromadb.PersistentClient(path=r"chroma")
# create collection
chroma_collection = chroma_client.create_collection("documents")

# create vector store on chromdb
chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
chroma_storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)


# indexing on nodes
chroma_index = VectorStoreIndex(nodes, storage_context=chroma_storage_context)


# Define your query
query_text = "Tell me about Saudi Arabia"

# Execute the query and get the response
response = chroma_query_engine.query(query_text)

# Print the response from the LLM
print(response.response)


