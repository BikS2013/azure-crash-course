from typing import List, Dict, Any, Tuple
from logger import log, INFO, ERROR, WARNING
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.documents import Document
import requests

class SearchEndPoint:
    endpoint: str
    key: str
    indexname: str
    api_version: str
    #embeddings: AzureOpenAIEmbeddings

    vector_store: AzureSearch

    def __init__(self, endpoint: str, key :str, indexname: str, api_version: str, 
                 #embeddings: AzureOpenAIEmbeddings
                ):
        self.endpoint = endpoint
        self.key = key
        self.indexname = indexname
        self.api_version = api_version
        #self.embeddings = embeddings

        self.vector_store = AzureSearch(
            azure_search_endpoint=self.endpoint,
            azure_search_key=self.key,
            index_name=self.indexname, 
            embedding_function=None
            #embedding_function=self.embeddings.embed_query,
        )
        
    def _get_headers(self): 
        return {
            'Content-Type': 'application/json',
            'api-key': self.key
        }
    
    def _get_index_url(self, update=False) -> str:
        if update:
            return f"{self.endpoint}/indexes/{self.indexname}?api-version={self.api_version}&allowIndexDowntime=true"
        else:
            return f"{self.endpoint}/indexes/{self.indexname}?api-version={self.api_version}"
    
    def _get_documents_url(self) -> str:
        return f"{self.endpoint}/indexes/{self.indexname}/docs/index?api-version={self.api_version}"
    
    def get_index_schema(self) -> Dict[str, Any]:
        response = requests.get(self._get_index_url(), headers=self._get_headers())

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            log(INFO, f"Index {self.indexname} does not exist")
            return None
        else:
            log(ERROR, f"Failed to get index schema : {response.text}")
            return None   
        
    def delete_index(self) -> bool:
        response = requests.delete(self._get_index_url(), headers=self._get_headers())
        if response.status_code in [200, 204, 404]:
            log(INFO, f"Index {self.indexname} deleted successfully or did not exist")
            return True
        else:
            log(ERROR, f"Failed to delete index: {response.text}")
            return False

    def create_index_schema(self, schema: dict) -> bool:
        response = requests.put(self._get_index_url(), headers=self._get_headers(), json=schema)
        if response.status_code in [200, 201]:
            log(INFO, f"Index schema created successfully {schema.AZURE_SEARCH_INDEX_NAME}")
            return True
        else:
            log(ERROR, f"Failed to create index schema: {response.text}")
            return False
          
    def update_index_schema(self, new_schema: dict) -> bool:
        response = requests.put(self._get_index_url(update=True), headers=self._get_headers(), json=new_schema)
        if response.status_code in [200, 201]:
            log(INFO, f"Index schema updated successfully {self.indexname}")
            return True
        else:
            log(ERROR, f"Failed to update index schema: {response.text}")
            return False

    def update_documents(self, data: List[Dict[str, Any]]) -> bool:
        response = requests.post(self._get_documents_url(), headers=self._get_headers(), json={ "value": data })
        if response.status_code in [200, 201]:
            return True
        else:
            log(ERROR, f"Failed to add document batch: {response.text}")
            return False

    def hybrid_search( self, query: str, top_k: int = 5, **kwargs: Any ) -> List[Tuple[Document, float]]:
        return self.vector_store.hybrid_search_with_relevance_scores(query, k=top_k, **kwargs)
    
    def vector_search( self, query: str, top_k: int = 5, **kwargs: Any ) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_relevance_scores(query, k=top_k, **kwargs)
