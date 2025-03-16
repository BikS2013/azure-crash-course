from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List
import requests 

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from logger import log, INFO, ERROR, WARNING
from search_endpoint import SearchEndPoint
from search_index import SearchIndex
from search_schema import schema

from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_SEARCH_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_CHAT_MODEL,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_KEY,
    DOMAINS,
)

class SearchManager :
    search_endpoint: SearchEndPoint
    chat_model: BaseChatModel
    domains: List[str]
    def __init__(self, search_endpoint: SearchEndPoint, chat_model: BaseChatModel):
        self.chat_model = chat_model
        self.search_endpoint = search_endpoint
        self.domains = DOMAINS


    def get_filter_conditions(self, unique_chunks: List[str], window: int = 3) -> str:
        """
        Generate filter conditions for unique titles
        """
        filter_conditions = []
        for chunk in unique_chunks:
            # Replace single quotes with double single quotes to escape them in OData
            escaped_chunk = chunk.replace("'", "''")
            for i in range(-window, window+1):
                filter_conditions.append(f"chunk_id eq '{str(int(escaped_chunk)+i)}'")
        
        return " or ".join(filter_conditions)
    
    def category_prompt(self) -> PromptTemplate:
        # Create prompt template
        from prompts import category_prompt_template
        return PromptTemplate(template=category_prompt_template, input_variables=['question'])
    
    def category_chain(self, question: str) -> str:
        """
        Get the category of the question
        """
        chain = self.category_prompt() | self.chat_model | StrOutputParser()
        result = chain.invoke({"question": question})
        return result 

    def search(self, query: str, top_k: int = 20, type_filter: str = "") -> List[Dict[str, Any]]:
        """
        Search documents using hybrid similarity search, pre filtering and then retrieve all the content of the unique documents
        """
        try:
            type_filter = self.category_chain(query)
            if type_filter in self.domains:
                # Perform similarity search
                results = self.search_endpoint.hybrid_search(query, top_k=top_k, filters=f"type eq '{type_filter}'")
            else:
                results = self.search_endpoint.hybrid_search(query, top_k=top_k)

            # Get unique chunks from results
            unique_chunks = list(set([res[0].metadata.get('chunk_id', '') for res in results]))            
            filter_expression = self.get_filter_conditions(unique_chunks)
            
            # Execute the search with the filter to get all the content of the documents
            results_all = self.search_endpoint.hybrid_search(query, top_k=100, filters=filter_expression)

            # Format results
            search_results = []
            for doc, score in results_all:
                result = {
                    'content': doc.page_content,
                    'source': doc.metadata.get('title', ''),
                    'score': score,
                    **{k: v for k, v in doc.metadata.items() if k != 'source'}
                }
                search_results.append(result)
            # log(INFO, f"Search completed successfully - Found {len(search_results)} results")
            return search_results

        except Exception as e:
            log(ERROR, f"Search error: {str(e)}")
            return []
    
def initialize_search_manager() -> tuple[SearchManager, BaseChatModel]:
    """Initialize Azure Search with OpenAI embeddings"""
    try:

        from config import(
            AZURE_TENANT_ID,
            AZURE_CLIENT_ID,
            AZURE_CLIENT_SECRET,
            AZURE_SUBSCRIPTION_ID,
            AZURE_RESOURCE_GROUP,
            AZURE_SEARCH_SERVICE_NAME,
            AZURE_INDEX_NAME,
            AZURE_INDEX_SEMANTIC_CONFIGURATION
        )

        from wrapper import Identity

        identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
        subscription = identity.get_subscription(subscription_id=AZURE_SUBSCRIPTION_ID)
        resource_group = subscription.get_resource_group(AZURE_RESOURCE_GROUP)

        search_service = subscription.get_search_service(AZURE_SEARCH_SERVICE_NAME)
        if search_service is None:
            raise ValueError(f"Search service '{AZURE_SEARCH_SERVICE_NAME}' not found in subscription '{AZURE_SUBSCRIPTION_ID}'")

        index =  search_service.get_index(AZURE_INDEX_NAME)

        result = index.perform_search(query_text="Ποιά είναι τα καταναλωτικά δανειακά προϊόντα?", highlight_fields="chunk", top=5)
        list = [item for item in result]
        
        exit(); 
        semantic_config_name = AZURE_INDEX_SEMANTIC_CONFIGURATION
        query="Ποιά είναι τα καταναλωτικά δανειακά προϊόντα?"
        if index is not None:
            result = index.perform_hybrid_search(query_text=query, 
                                            #query_vector=query_embedding, 
                                            vector_fields="text_vector", 
                                            use_semantic_search=True,
                                            semantic_config_name=semantic_config_name)

        exit()

        location = SearchEndPoint(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME, AZURE_SEARCH_API_VERSION, 
                                  #embeddings
                                  )

        index = SearchIndex(location, AZURE_SEARCH_INDEX_NAME, schema )
        

        llm = AzureChatOpenAI(
            model=AZURE_OPENAI_CHAT_MODEL,
            temperature=0.0,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )
                
        log(INFO, f"Successfully initialized AzureSearchManager with endpoint: {AZURE_SEARCH_ENDPOINT}")
        
        import azure.search.documents.indexes.models as azsdim        
        def get_index_fields():
            # Define the index fields
            fields = [
                azsdim.SimpleField(name="chunk_id", type=azsdim.SearchFieldDataType.String, key=True),
                azsdim.SearchableField(name="chunck", type=azsdim.SearchFieldDataType.String, analyzer_name="en.lucene"),
                azsdim.SearchableField(name="title", type=azsdim.SearchFieldDataType.String, analyzer_name="en.lucene"),
                azsdim.SimpleField(name="text_vector", type=azsdim.SearchFieldDataType.Collection(azsdim.SearchFieldDataType.Single),
                                vector_search_dimensions=1536,
                                vector_search_profile_name="default-profile"),
                azsdim.SearchableField(name="url", type=azsdim.SearchFieldDataType.String, analyzer_name="en.lucene"),
                azsdim.SearchableField(name="name", type=azsdim.SearchFieldDataType.String, analyzer_name="en.lucene"),
            ]
            return fields
        
        index = azsdim.SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )        



        manager = SearchManager(location, index, 
                                #embed_document, 
                                llm)
        return manager

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error connecting to Azure Search: {str(e)}"
        log(ERROR, error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        log(ERROR, f"Error initializing AzureSearchManager: {str(e)}")
        raise

if __name__ == "__main__":
    manager= initialize_search_manager()
