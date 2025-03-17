from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict, List
import requests 

from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from logger import log, INFO, ERROR, WARNING
from tenacity import retry, stop_after_attempt, wait_random_exponential


def search_manager() -> List[Dict]:
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
            AZURE_INDEX_SEMANTIC_CONFIGURATION,
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_EMBEDDING_MODEL,
            AZURE_OPENAI_API_VERSION,
            AZURE_OPENAI_KEY,
        )
        from openai import AzureOpenAI
        def get_embedding_client( api_key: str, api_base: str, api_version: str) : 
            client = AzureOpenAI(
                api_key=api_key,  # Set this as an environment variable
                api_version=api_version,           # Using a recent API version
                azure_endpoint=api_base,  # e.g. "https://your-resource.openai.azure.com/"
            )
            return client

        @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
        def generate_embeddings(text: str, openai_client, model: str = "text-embedding-3-small" ) -> List[float]:
            try : 
                model_deployed = model
                response = openai_client.embeddings.create( input=[text],model="text-embedding-3-small" )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error: {str(e)}")
                raise e 
   
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
        
        semantic_config_name = AZURE_INDEX_SEMANTIC_CONFIGURATION
        query="Ποιά είναι τα καταναλωτικά δανειακά προϊόντα?"
        openai_client = get_embedding_client(AZURE_OPENAI_KEY, api_base=AZURE_OPENAI_ENDPOINT, api_version=AZURE_OPENAI_API_VERSION )
        query_embedding = generate_embeddings(query, openai_client, model=AZURE_OPENAI_EMBEDDING_MODEL)
        if index is not None:
            result = index.search_with_context_window(
                            query_text=query,
                            query_vector=query_embedding,
                            vector_fields="text_vector",
                            use_semantic_search=True,
                            semantic_config_name=semantic_config_name,
                            window_size=3,  # 3 chunks before and after
                            top=5,
                            )

        return result
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error connecting to Azure Search: {str(e)}"
        log(ERROR, error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        log(ERROR, f"Error initializing AzureSearchManager: {str(e)}")
        raise

if __name__ == "__main__":
    result = search_manager()
