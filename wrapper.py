from typing import Any, List, Dict, Optional, Union
import os
from dotenv import load_dotenv
load_dotenv()

from azure.identity import ClientSecretCredential
from azure.mgmt.resource import SubscriptionClient, ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
import azure.mgmt.storage.models as azstm
from azure.mgmt.storage.models import StorageAccount
import azure.mgmt.resource.subscriptions.models as azsbm
from azure.mgmt.search import SearchManagementClient
import azure.mgmt.search.models as azsrm
import azure.search.documents as azsd
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery
from tenacity import retry, stop_after_attempt, wait_random_exponential
class Identity:

    tenant_id: str
    subscription_id: str
    client_id: str
    client_secret: str
    credential: ClientSecretCredential

    subscription_client: SubscriptionClient

    def __init__(self, tenant_id, subscription_id, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.subscription_id = subscription_id       
        self.credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
        self.subscription_client = SubscriptionClient(self.credential)
    
    def get_credential(self):
        return self.credential
    
    def get_subscriptions(self) -> list[azsbm.Subscription]:
        subscriptions = list(self.subscription_client.subscriptions.list())
        return subscriptions 
    
    def get_subscription(self, subscription_id) -> "Subscription":
        for sub in self.get_subscriptions():
            if sub.subscription_id == subscription_id:
                return Subscription(self, sub, sub.subscription_id)
        return None
        
class Subscription:

    identity: Identity
    subscription: azsbm.Subscription
    subscription_id: str

    resource_client: ResourceManagementClient
    storage_client: StorageManagementClient

    def __init__(self, identity: Identity, 
                 subscription: azsbm.Subscription, subscription_id: str):
        self.identity = identity
        self.subscription = subscription
        self.subscription_id = subscription_id
        self.resource_client = ResourceManagementClient(self.identity.get_credential(), self.subscription_id)
        self.storage_client = StorageManagementClient(self.identity.get_credential(), self.subscription_id)

    def get_resource_group(self, group_name: str) -> "ResourceGroup":
        groups = self.resource_client.resource_groups.list()
        for group in groups:
            if group.name.lower() == group_name.lower():
                return ResourceGroup(self, group)
        return None
    
    def create_resource_group(self, group_name: str, location: str) -> "ResourceGroup":
        result = self.resource_client.resource_groups.create_or_update(
            group_name,
            {"location": location}
        )
        if result is not None:
            return ResourceGroup(self, result)
        return None
    
    def get_search_sevices(self) -> list[azsrm.SearchService]:
        search_mgmt_client = SearchManagementClient(self.identity.get_credential(), self.subscription_id)
        services = list(search_mgmt_client.services.list_by_subscription())
        return services 
    
    def get_search_service(self, service_name: str) -> "SearchService":
        services = self.get_search_sevices()
        for service in services:
            if service.name == service_name:
                resource_group_name = service.id.split("/")[4] 
                resource_group = self.get_resource_group(resource_group_name)
                return SearchService(resource_group, service)
        return None
    
    def get_storage_management_client(self) -> StorageManagementClient:
        if self.storage_client is None:
            self.storage_client = StorageManagementClient(self.identity.get_credential(), self.subscription_id) 
        return self.storage_client
    
    def get_storage_accounts(self) -> list[azstm.StorageAccount]:
        accounts = list(self.storage_client.storage_accounts.list())
        return accounts
    
                
import azure.mgmt.resource.resources.models as azrm

class ResourceGroup:
    azure_resource_group: azrm.ResourceGroup
    subscription: Subscription

    def __init__(self, subscription: Subscription, azure_resource_group ):
        self.subscription = subscription
        self.azure_resource_group = azure_resource_group

    def get_name(self) -> str:
        return self.azure_resource_group.name

    def get_resources(self) -> list[azrm.GenericResource]:
        resources = self.subscription.resource_client.resources.list_by_resource_group(self.azure_resource_group.name)
        return resources
    
    def get_storage_management_client(self) -> StorageManagementClient:
        return self.subscription.get_storage_management_client()
    
    def create_search_service(self, name: str, location: str) -> "SearchService":
        search_mgmt_client = SearchManagementClient(self.subscription.identity.get_credential(), 
                                                    self.subscription.subscription_id)
        # Define the search service
        search_service = {
            "location": location,
            "sku": {
                "name": "basic"  # Options: free, basic, standard, standard2, standard3, storage_optimized_l1, storage_optimized_l2
            }
        }
        operation = search_mgmt_client.services.begin_create_or_update(
            resource_group_name=self.azure_resource_group.name,
            search_service_name=name,
            service=search_service
        )
        search_service = operation.result()
        return SearchService(self, search_service)
    
    def get_storage_account(self, account_name: str) -> azstm.StorageAccount:
        storage_client = self.subscription.get_storage_management_client()
        try:
            account = storage_client.storage_accounts.get_properties(resource_group_name = self.azure_resource_group.name, 
                                                                account_name = account_name)
        except Exception as e:
            print(f"Error at ResourceGroup.get_storage_account(): {str(e)}")
            account = None
        return account    

    def create_storage_account(self, account_name: str, location: str) -> azstm.StorageAccount:
        storage_client = self.subscription.get_storage_management_client()
        params = azstm.StorageAccountCreateParameters(
            sku=azstm.Sku(name="Standard_LRS"), 
            kind=azstm.Kind.STORAGE_V2, 
            location=location
        )
        result = storage_client.storage_accounts.begin_create(resource_group_name=self.azure_resource_group.name, 
                                                              account_name=account_name, 
                                                              parameters=params)
        return result.result()

from azure.storage.blob import BlobServiceClient
class StorageAccount: 
    storage_account: azstm.StorageAccount
    resource_group: ResourceGroup

    storage_key: str
    connection_string_description: str

    def __init__(self, resource_group: ResourceGroup, storage_account: azstm.StorageAccount):
        self.resource_group = resource_group
        self.storage_account = storage_account
        client = resource_group.get_storage_management_client()
        keys = client.storage_accounts.list_keys(resource_group_name=resource_group.get_name(), 
                                                 account_name=storage_account.name)
        self.storage_key = keys.keys[0].value
        self.connection_string_description = f"DefaultEndpointsProtocol=https;AccountName={storage_account.name};AccountKey={self.storage_key};EndpointSuffix=core.windows.net"
    
    def get_name(self) -> str:
        return self.storage_account.name
    
    def get_blob_service_client(self) -> BlobServiceClient:
        return BlobServiceClient.from_connection_string(self.connection_string_description) 
    
    def get_container_client(self, container_name: str):
        client = self.get_blob_service_client()
        container_client = client.get_container_client(container_name)
        

    


import azure.search.documents.indexes as azsdi
import azure.search.documents.indexes.models as azsdim
from azure.core.credentials import AzureKeyCredential

class SearchService:
    search_service: azsrm.SearchService
    resource_group: ResourceGroup
    index_client: azsdi.SearchIndexClient
    search_client: azsd.SearchClient
    openai_client: Any

    def __init__(self, resource_group: ResourceGroup, search_service: azsrm.SearchService):
        self.resource_group = resource_group
        self.search_service = search_service
        self.index_client = None
        self.search_client = None
        self.openai_client = None
        self.index_name = os.getenv("INDEX_NAME", "default-index")
    
    def get_admin_key(self) -> str:
        search_mgmt_client = SearchManagementClient(self.resource_group.subscription.identity.get_credential(),
                                                    self.resource_group.subscription.subscription_id)
        keys = search_mgmt_client.admin_keys.get(resource_group_name=self.resource_group.azure_resource_group.name,
                                                search_service_name=self.search_service.name)
        return keys.primary_key

    def get_credential(self) -> AzureKeyCredential:
        return AzureKeyCredential(self.get_admin_key())

    def get_service_endpoint(self) -> str:
        return f"https://{self.search_service.name}.search.windows.net"

    def get_index_client(self) -> azsdi.SearchIndexClient:
        if self.index_client is None:
            self.index_client = azsdi.SearchIndexClient(self.get_service_endpoint(),
                                                         self.get_credential())
        return self.index_client
    
    def get_indexes(self) -> List[azsdim.SearchIndex]:
        index_client = self.get_index_client()
        indexes = list(index_client.list_indexes())
        return indexes

    def get_index(self, index_name: str) -> "SearchIndex":
        indexes = self.get_indexes()
        for index in indexes:
            if index.name == index_name:
                return SearchIndex(self, index.name, index.fields, index.vector_search)
        return None
      
    def create_or_update_index(self, index_name: str, fields: List[azsdim.SearchField])->"SearchIndex":
        return SearchIndex(self, index_name, fields)
    
    
    def add_semantic_configuration(self,
                                  title_field: str = "title",
                                  content_fields: List[str] = None,
                                  keyword_fields: List[str] = None,
                                  semantic_config_name: str = "default-semantic-config"):
        """
        Add semantic configuration to the index.
        
        Args:
            title_field: The name of the title field
            content_fields: List of content fields to prioritize
            keyword_fields: List of keyword fields to prioritize
            semantic_config_name: The name of the semantic configuration
            
        Returns:
            The updated index
        """
        if content_fields is None:
            content_fields = ["content"]
        
        if keyword_fields is None:
            keyword_fields = ["tags"]
        
        # Get the existing index
        index = self.get_index_client().get_index(self.index_name)
        
        # Define semantic configuration
        semantic_config = azsdim.SemanticConfiguration(
            name=semantic_config_name,
            prioritized_fields=azsdim.PrioritizedFields(
                title_field=azsdim.SemanticField(field_name=title_field),
                prioritized_content_fields=[
                    azsdim.SemanticField(field_name=field) for field in content_fields
                ],
                prioritized_keywords_fields=[
                    azsdim.SemanticField(field_name=field) for field in keyword_fields
                ]
            )
        )
        
        # Create semantic settings with the configuration
        semantic_settings = azsdim.SemanticSettings(
            configurations=[semantic_config]
        )
        
        # Add semantic settings to the index
        index.semantic_settings = semantic_settings
        
        # Update the index
        result = self.get_index_client().create_or_update_index(index)
        return result
class SearchIndex:
    index_name: str
    fields: List[azsdim.SearchField]
    vector_search: azsdim.VectorSearch
    search_service: SearchService
    azure_index: azsdim.SearchIndex

    def __init__(self, search_service: SearchService, index_name: str, fields: List[azsdim.SearchField], vector_search: azsdim.VectorSearch):
        self.search_service = search_service
        self.index_name = index_name
        self.fields = fields
        self.vector_search = vector_search

        # SimpleField, SearchableField, ComplexField, are derived from SearchField
        index_definition = azsdim.SearchIndex(name=self.index_name, fields=fields)
        self.azure_index = self.search_service.get_index_client().create_or_update_index(index_definition)


    def get_search_client(self, index_name: Optional[str] = None) -> azsd.SearchClient:

        search_client = self.search_service.search_client            
        if search_client is None or search_client.index_name != self.index_name:
            search_client = azsd.SearchClient(
                endpoint=self.search_service.get_service_endpoint(),
                index_name=self.index_name,
                credential=self.search_service.get_credential()
            )
        return search_client

    def perform_hybrid_search(self,
                             query_text: str,
                             query_vector: List[float],
                             vector_fields: str = None,
                             search_options: Dict[str, Any] = None,
                             use_semantic_search: bool = False,
                             semantic_config_name: str = "default-semantic-config") -> List[Dict[str, Any]]:
        """
        Perform a hybrid search combining traditional keyword search with vector search.
        Args:
            query_text: The search query text
            vector_fields: List of fields to perform vector search on (default: ["content_vector"])
            search_options: Additional search options
            use_semantic_search: Whether to use semantic search capabilities
            semantic_config_name: The name of the semantic configuration to use
        Returns:
            A list of search results
        """
        # Default vector fields if not provided
        if vector_fields is None:
            vector_fields = "text_vector"
        
        # Create vectorized query
        vectorized_query = VectorizedQuery(vector=query_vector, k=50, fields=vector_fields)
        #vectorized_query = VectorizableTextQuery(text=query, k=50, fields=vector_fields)
        
        # Default search options
        default_options = {
            "search_text": query_text,  # Traditional keyword search
            "vector_queries": [vectorized_query],  # Vector search component
            "top": 10,
            "select": "*",
            "include_total_count": True,
        }
        
        # Add semantic search if requested
        if use_semantic_search:
            default_options.update({
                "query_type": "semantic",
                "semantic_configuration_name": semantic_config_name,
                "query_caption": "extractive", 
                "query_answer": "extractive",
            })
        
        # Update with any user-provided options
        if search_options:
            default_options.update(search_options)
        
        # Execute the search
        search_client = self.get_search_client()
        results = search_client.search(**default_options)
        
        # Process and return the results
        processed_results = []
        for result in results:
            processed_result = dict(result)
            processed_results.append(processed_result)
        
        return processed_results
    


from openai import AzureOpenAI
def get_embedding_client( api_key: str, api_base: str, api_version: str) : 
    client = AzureOpenAI(
        api_key=api_key,  # Set this as an environment variable
        api_version=api_version,           # Using a recent API version
        azure_endpoint=api_base  # e.g. "https://your-resource.openai.azure.com/"
    )
    return client

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text: str, openai_client, model: str = "text-embedding-3-small" ) -> List[float]:
    try : 
        response = openai_client.embeddings.create( model=model, input=text )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: {str(e)}")
        raise e 
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    tenant_id = os.getenv("AZURE_TENANT_ID")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")
    
    group_name = os.getenv("RESOURCE_GROUP_NAME")
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    location = os.getenv("RESOURCE_LOCATION")
    index_name = os.getenv("INDEX_NAME")
    
    # For OpenAI API (you would need to add these to your .env file)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_version = os.getenv("AZURE_API_VERSION")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_embedding_model = os.getenv("AZURE_EMBEDDING_MODEL")

    identity = Identity(tenant_id, subscription_id, client_id, client_secret)
    subscription = identity.get_subscription(subscription_id=subscription_id)
    resource_group = subscription.get_resource_group(group_name)

    


    query = "Ποιές είναι οι διαφορές στη διαδικασία έκδοσης χρεωστικών και πιστωτικών καρτών;"
    openai_client = get_embedding_client(azure_openai_api_key, api_base=azure_openai_endpoint, api_version=azure_api_version )
    query_embedding = generate_embeddings(query, openai_client, model=azure_embedding_model)


    storage_accounts = subscription.get_storage_accounts()


    storage_account = resource_group.get_storage_account(storage_account_name)

    create_result = resource_group.create_storage_account(storage_account_name + "a1", location)
    
    print("Getting or creating search service...")
    search_service_name = "athenamar2025"
    search_service = subscription.get_search_service(search_service_name)
    if search_service is None:
        print(f"Creating search service '{search_service_name}'...")
        search_service = resource_group.create_search_service(search_service_name, location)

    index =  search_service.get_index(index_name)
    
    semantic_config_name = "athena-vector-mar2025-semantic-configuration"
    if index is not None:
        result = index.perform_hybrid_search(query_text=query, 
                                        query_vector=query_embedding, 
                                        vector_fields="text_vector", 
                                        use_semantic_search=True,
                                        semantic_config_name=semantic_config_name)
        

    print(f"Search service endpoint: {search_service.get_service_endpoint()}")
    
    # Define vector search configuration
    vector_search = azsdim.VectorSearch(
        algorithms=[
            azsdim.VectorSearchAlgorithmConfiguration(
                name="default-algorithm",
                kind="hnsw",
                hnsw_parameters=azsdim.HnswParameters(
                    m=4,  # Number of connections per node
                    ef_construction=400,  # Size of the dynamic list for nearest neighbors
                    ef_search=500,  # Size of the dynamic list for searching
                    metric="cosine"  # Distance metric (cosine, euclidean, dotProduct)
                )
            )
        ],
        profiles=[
            azsdim.VectorSearchProfile(
                name="default-profile",
                algorithm_configuration_name="default-algorithm"
            )
        ]
    )
    
    # Define the index fields
    fields = [
        azsdim.SimpleField(name="id", type=azsdim.SearchFieldDataType.String, key=True),
        azsdim.SearchableField(name="title", type=azsdim.SearchFieldDataType.String, analyzer_name="en.lucene"),
        azsdim.SimpleField(name="title_vector", type=azsdim.SearchFieldDataType.Collection(azsdim.SearchFieldDataType.Single),
                           vector_search_dimensions=1536,
                           vector_search_profile_name="default-profile"),
        azsdim.SearchableField(name="content", type=azsdim.SearchFieldDataType.String, analyzer_name="en.lucene"),
        azsdim.SimpleField(name="content_vector", type=azsdim.SearchFieldDataType.Collection(azsdim.SearchFieldDataType.Single),
                           vector_search_dimensions=1536,
                           vector_search_profile_name="default-profile"),
        azsdim.SearchableField(name="category", type=azsdim.SearchFieldDataType.String, filterable=True, facetable=True),
        azsdim.SimpleField(name="timestamp", type=azsdim.SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True)
    ]
    
    # Create the index with vector search configuration
    print(f"Creating or updating index '{index_name}'...")
    index = azsdim.SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    try:
        result = search_service.get_index_client().create_or_update_index(index)
        print(f"Index '{index_name}' created or updated successfully.")
        
        # Add semantic configuration
        print("Adding semantic configuration...")
        search_service.add_semantic_configuration(
            title_field="title",
            content_fields=["content"],
            keyword_fields=["category"]
        )
        print("Semantic configuration added successfully.")
        
        # Example of using hybrid search (if OpenAI API key is available)
        if openai_api_key:
            print("\nSetting up OpenAI client...")
            search_service.setup_openai_client(openai_api_key)
            
            print("\nPerforming hybrid search...")
            query = "Azure AI Search implementation techniques"
            results = search_service.perform_hybrid_search(
                query_text=query,
                vector_fields=["content_vector"],
                use_semantic_search=True
            )
            
            print(f"\nHybrid search results for: '{query}'")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                print(f"   ID: {result.get('id', 'No ID')}")
                print(f"   Score: {result.get('@search.score', 'No score')}")
                
                # Display highlights if available
                if '@search.highlights' in result and 'content' in result['@search.highlights']:
                    print("   Highlights:")
                    for highlight in result['@search.highlights']['content']:
                        print(f"   ...{highlight}...")
            
            print(f"\nFound {len(results)} documents")
        elif azure_openai_api_key and azure_openai_endpoint and azure_openai_deployment:
            print("\nSetting up Azure OpenAI client...")
            search_service.setup_openai_client(
                api_key=azure_openai_api_key,
                api_base=azure_openai_endpoint,
                api_version="2023-05-15"
            )
            
            print("\nPerforming hybrid search with Azure OpenAI...")
            query = "Azure AI Search implementation techniques"
            results = search_service.perform_hybrid_search(
                query_text=query,
                vector_fields=["content_vector"],
                use_semantic_search=True,
                deployment_name=azure_openai_deployment
            )
            
            print(f"\nHybrid search results for: '{query}'")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                print(f"   ID: {result.get('id', 'No ID')}")
                print(f"   Score: {result.get('@search.score', 'No score')}")
        else:
            print("\nNo OpenAI API key provided. Skipping hybrid search demonstration.")
            print("To use hybrid search, add OPENAI_API_KEY to your .env file.")
            print("Or for Azure OpenAI, add AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
