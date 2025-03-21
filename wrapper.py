import os
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from time import time 
from typing import Any, Callable, List, Dict, Optional, Tuple, Union

from azure.identity import ClientSecretCredential
from azure.mgmt.resource import SubscriptionClient, ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
import azure.mgmt.storage.models as azstm
from azure.mgmt.storage.models import StorageAccount
import azure.mgmt.resource.subscriptions.models as azsbm
from azure.mgmt.search import SearchManagementClient
import azure.mgmt.search.models as azsrm
import azure.search.documents as azsd
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery, QueryType
from tenacity import retry, stop_after_attempt, wait_random_exponential
class Identity:

    tenant_id: str
    client_id: str
    client_secret: str
    credential: ClientSecretCredential

    subscription_client: SubscriptionClient

    def __init__(self, tenant_id, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
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

from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

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
    
    def get_cognitive_client(self) -> CognitiveServicesManagementClient: 
        cognitive_client: CognitiveServicesManagementClient 
        cognitive_client = CognitiveServicesManagementClient(
            credential=self.identity.get_credential(), 
            subscription_id=self.subscription_id
        )
        return cognitive_client
                
import azure.mgmt.resource.resources.models as azrm
import azure.mgmt.cognitiveservices.models as azcsm
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
    
    def get_storage_account(self, account_name: str) -> "StorageAccount":
        storage_client = self.subscription.get_storage_management_client()
        try:
            account = storage_client.storage_accounts.get_properties(resource_group_name = self.azure_resource_group.name, 
                                                                account_name = account_name)
        except Exception as e:
            print(f"Error at ResourceGroup.get_storage_account(): {str(e)}")
            account = None
        if account is not None:
            return StorageAccount(self, account)
        else:
            return None

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

    def get_ai_service(self, service_name:str) -> "AIService":
        cognitive_client = self.subscription.get_cognitive_client()
        cognitive_accounts = cognitive_client.accounts.list_by_resource_group(self.azure_resource_group.name)

        accounts = [account for account in cognitive_accounts]
        openai_services_names = [ account.name for account in  accounts]

        for account in accounts :
            if account.kind.lower() == "openai" and account.name.lower() == service_name.lower() :
                return AIService(self, cognitive_client=cognitive_client, azure_Account=account)
        return None
    

        
from azure.storage.blob import BlobServiceClient, ContainerProperties, ContainerClient, BlobProperties
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

    def get_containers(self) -> list[ContainerProperties]:
        client = self.get_blob_service_client()
        containers = client.list_containers()
        return [container for container in containers]

    def get_container(self, container_name:str) -> "Container":
        client = self.get_blob_service_client()
        container = client.get_container_client(container_name)
        return Container(self, container)
class Container:
    container_client: ContainerClient
    storage_account: StorageAccount

    def __init__(self, storage_account: StorageAccount, container_client: ContainerClient):
        self.storage_account = storage_account
        self.container_client = container_client

    def get_blob_names(self) -> list[str]:
        blobs = self.container_client.list_blob_names()
        return [blob for blob in blobs]
    
    def get_blobs(self) -> list[BlobProperties]:
        blobs = self.container_client.list_blobs()
        return [blob for blob in blobs]

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
        index_client = self.get_index_client()
        index = index_client.get_index(index_name)
        if index and index.name == index_name:
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

def get_std_vector_search( connections_per_node:int = 4, 
                          neighbors_list_size: int = 400, 
                          search_list_size: int = 500, 
                          metric: str = "cosine") -> azsdim.VectorSearch:
    """
    Get a standard vector search configuration.
    Args:
        connections_per_node: Number of connections per node. Default is 4.
        neighbors_list_size: Size of the dynamic list for nearest neighbors. Default is 400.
        search_list_size: Size of the dynamic list for searching. Default is 500.
        metric: Distance metric (cosine, euclidean, dotProduct). Default is cosine.
    Returns:
        A vector search configuration
    """
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
    return vector_search
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

    def get_search_client(self, index_name: Optional[str] = None ) -> azsd.SearchClient:
        if not index_name:
            index_name = self.index_name
        search_client = self.search_service.search_client            
        if search_client is None or search_client.index_name != index_name:
            search_client = azsd.SearchClient(
                endpoint=self.search_service.get_service_endpoint(),
                index_name=index_name,
                credential=self.search_service.get_credential()
            )
        return search_client  

    def extend_index_schema( self, new_fields: List[azsdim.SearchField] ) -> bool | None :
        """
        Extend an Azure AI Search index schema with new fields
        
        Args:
            new_fields: List of new field definitions to add
            Indicative fields:
            azsdim.SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            azsdim.SimpleField(name="publication_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            azsdim.SearchableField(name="summary", type=SearchFieldDataType.String, analyzer_name="en.microsoft")
        """

        try:
            existing_field_names = [field.name for field in self.azure_index.fields]
            
            # Filter out any fields that already exist in the index
            fields_to_add = [field for field in new_fields if field.name not in existing_field_names]
            
            if not fields_to_add:
                print("No new fields to add - all specified fields already exist in the index")
                return True
                
            self.azure_index.fields.extend(fields_to_add)
            
            index_client: azsdi.SearchIndexClient = self.search_service.get_index_client()
            result : bool | None = index_client.create_or_update_index(self.azure_index)
            print(f"Successfully extended index '{self.index_name}' with {len(fields_to_add)} new fields")
            
            # Return the updated index
            return result
            
        except Exception as e:
            print(f"Error extending index: {str(e)}")
            raise

    def process_data_in_batches( self, 
                                index_name: str,
                                transaction: Callable[[List[Dict[str, Any]]], int],
                                search_text: str = "*",
                                batch_size: int = 100 ) -> Tuple[int, int]:
        '''
        Process data in batches from an Azure AI Search index

        Args:
            index_name: Name of the index to process. If None, the current index is used. 
            transaction: Function to process a batch of documents. Gets a list of documents and returns the number of successful transactions. The transaction function could upload documents to the same or another index.
            batch_size: Number of documents to process in each batch
        
        Returns:
            Tuple of (succeeded_count, document_count)

        '''
        if index_name is None:
            index_name = self.index_name
        search_client = self.get_search_client(index_name)
        skip = 0
        document_count = 0
        succeeded_count = 0 
        
        while True:
            results = search_client.search( search_text=search_text, include_total_count=True, skip=skip, top=batch_size )
            
            # Get the total document count
            if document_count == 0:
                document_count = results.get_count()
                print(f"Retrieved {document_count} documents to process")
            
            documents_to_process = []
            batch_documents = list(results)
            if not batch_documents:
                break  # No more documents to process
            
            for doc in batch_documents:
                documents_to_process.append(doc)
            
            # Upload the batch to the target index
            if documents_to_process:
                transaction_result = transaction(documents_to_process)
                
                succeeded_count += transaction_result
                print(f"Processed batch: {transaction_result}/{len(documents_to_process)} documents (offset: {skip})")
            
            # Move to the next batch
            skip += batch_size
            
            # Check if we've processed all documents
            if skip >= document_count:
                break
        print(f"Successfully processed {succeeded_count}/{document_count} documents from  index '{index_name}'")
        return (succeeded_count, document_count)

    def copy_index_data( self, source_index_name: str, target_index_name: str, fields_to_copy:List[str], batch_size:int=100) -> Tuple[int, int]:
        """
        Copy data from source index to target index, excluding the removed fields
        
        Args:
            source_index_name: Name of the source index. if None the current index is used
            target_index_name: Name of the target index. if None the current index is used
            fields_to_copy: List of field names to copy from source to target. If None, all fields are copied.
            batch_size: Number of documents to process in each batch
        Returns:
            Tuple of (succeeded_count, document_count)
        """
        if not source_index_name:
            source_index_name = self.index_name
        if not target_index_name:
            target_index_name = self.index_name
        if source_index_name == target_index_name:
            print("Source and target index names are the same. No action taken.")
            return (0, 0)
        
        target_client = self.get_search_client(target_index_name)
        
        def copy_and_upload_documents(documents: List[Dict[str, Any]]) -> int:
            documents_to_upload = []
            for doc in documents:
                # Create a new document with the selected fields
                new_doc = {key: value for key, value in doc.items() if not fields_to_copy or (key in fields_to_copy) }
                documents_to_upload.append(new_doc)            
            # Upload the batch to the target index
            if documents_to_upload:
                result = target_client.upload_documents(documents=documents_to_upload)
                
                succeeded = sum(1 for r in result if r.succeeded)

            return succeeded

        result = self.process_data_in_batches(index_name = source_index_name, transaction=copy_and_upload_documents)
        return result

    def copy_index_structure( self, fields_to_copy:List[str]=None, new_index_name=None ) -> azsdim.SearchIndex:
        """
        Make a copy of an Azure AI Search index with a subset of fields.
        
        Args:
            fields_to_remove: List of field names to copy/replicate. if None, all fields are copied.
            new_index_name: Name for the new index (defaults to original_index_name + "_new")
        
        Returns:
            str: Name of the new index
        """

        try:
            original_index = self.azure_index
            if not new_index_name:
                new_index_name = f"{self.index_name}_new"

            new_fields = [field for field in original_index.fields if not fields_to_copy or field.name in fields_to_copy]
            
            if len(new_fields) == 0:
                print(f"None of the specified fields exist in index '{self.index_name}'")
                return self.index_name
            
            # Create a new index definition
            if hasattr(original_index, "semantic_settings"):
                semantic_settings = original_index.semantic_settings
            else:
                semantic_settings = None
                
            new_index = azsdim.SearchIndex(
                name=new_index_name,
                fields=new_fields,
                # Copy other index properties that might be important
                scoring_profiles=original_index.scoring_profiles,
                default_scoring_profile=original_index.default_scoring_profile,
                cors_options=original_index.cors_options,
                suggesters=original_index.suggesters,
                analyzers=original_index.analyzers,
                tokenizers=original_index.tokenizers,
                token_filters=original_index.token_filters,
                char_filters=original_index.char_filters,
                semantic_settings=semantic_settings,
                vector_search=original_index.vector_search
            )
            
            # Create the new index
            result = self.search_service.get_index_client().create_or_update_index(new_index)
            
            return result
            
        except Exception as e:
            print(f"Error copying index structure: {str(e)}")
            raise

    def perform_search(self, fields_to_select:str="*", highlight_fields:str="chunk", filter_expression:str=None, top:int=10,
                       query_text:str=None, search_options:Dict[str, Any]=None) -> azsd.SearchItemPaged[Dict]:
        search_options = {
            "include_total_count": True,
            "select": fields_to_select,
            "highlight_fields": highlight_fields,
            "highlight_pre_tag": "<b>",
            "highlight_post_tag": "</b>"
        }
        if filter_expression:
            search_options["filter"] = filter_expression
        if top:
            search_options["top"] = top
        search_client = self.get_search_client()
        results = search_client.search(query_text, **search_options)
        return results
    
    def get_adjacent_chunks(self, all_chunks: List[Dict]) -> List[Dict]:
        # Organize chunks by parent_id
        parent_chunks = defaultdict(list)
        for chunk in all_chunks:
            if 'parent_id' in chunk and 'chunk_id' in chunk:
                parent_chunks[chunk['parent_id']].append(chunk)
        
        all_chunks_by_parent = {}
        chunk_position_map = {}
        # Sort chunks within each parent document and create position maps
        for parent_id, chunks in parent_chunks.items():
            # Try to sort chunks within each parent document
            try:
                # First try to sort by chunk_id if it contains a number
                def extract_number(chunk):
                    chunk_id = chunk.get('chunk_id', '')
                    # Try to extract a number from the chunk_id
                    if '_' in chunk_id:
                        parts = chunk_id.split('_')
                        if parts[-1].isdigit():
                            return int(parts[-1])
                    if chunk_id.isdigit():
                        return int(chunk_id)
                    return chunk_id  # Fall back to string sorting
                
                sorted_chunks = sorted(chunks, key=extract_number)
            except:
                # If sorting fails, keep the original order
                sorted_chunks = chunks
            
            # Store the sorted chunks
            all_chunks_by_parent[parent_id] = sorted_chunks
            
            # Create a position map for quick lookup
            for i, chunk in enumerate(sorted_chunks):
                chunk_position_map[(parent_id, chunk['chunk_id'])] = i
            
        return all_chunks_by_parent, chunk_position_map

    def search_with_context_window(self,
                             query_text: str,
                             query_vector: List[float],
                             vector_fields: str = None,
                             search_options: Dict[str, Any] = None,
                             use_semantic_search: bool = False,
                             semantic_config_name: str = "default-semantic-config",
                             top: int = 10,
                             window_size=3):
        """
        Perform a search and retrieve a window of context chunks around each result.
        
        Args:
            query_text (str): The search query text
            window_size (int): Number of chunks to include before and after each result
            semantic_enabled (bool): Whether to enable semantic search
            top_k (int): Number of results to return
            vector_fields (list): List of vector fields to search
            text_fields (list): List of text fields to search
            filter_condition (str): Optional OData filter condition
            
        Returns:
            list: Search results enriched with context windows
        """
        # Get initial search results
        results = self.perform_hybrid_search(query_text=query_text, 
                                        query_vector=query_vector, 
                                        vector_fields=vector_fields, 
                                        use_semantic_search=use_semantic_search,
                                        top=top,
                                        semantic_config_name=semantic_config_name)
        
        if not results:
            return []
        
        # Collect all parent_ids from the results
        parent_ids = set()
        for result in results:
            if 'parent_id' in result:
                parent_ids.add(result['parent_id'])
        
        # Batch retrieve all chunks for all parent documents
        if parent_ids:
            # Create a filter to get all chunks from all parent documents in one query
            parent_filter = " or ".join([f"parent_id eq '{pid}'" for pid in parent_ids])
            
            search_options = {
                "include_total_count": True,
                "select": "*"
            }
            # Retrieve all chunks (up to a reasonable limit)
            all_chunks = list(self.perform_search("*", 
                                                  filter_expression=parent_filter,
                                                  top=1000,  # Adjust based on your needs
                                                  search_options=search_options))
            
            # Group chunks by parent_id
            all_chunks_by_parent, chunk_position_map = self.get_adjacent_chunks(all_chunks)
        
        # Enrich results with context windows
        enriched_results = []
        for result in results:
            parent_id = result.get('parent_id')
            chunk_id = result.get('chunk_id')
            
            # Initialize empty context window
            context_window = {
                'before': [],
                'after': []
            }
            
            if parent_id and chunk_id and parent_id in all_chunks_by_parent:
                parent_chunks = all_chunks_by_parent[parent_id]
                
                # Find the position of this chunk
                position = chunk_position_map.get((parent_id, chunk_id))
                
                if position is not None:
                    # Get previous chunks (up to window_size)
                    start_idx = max(0, position - window_size)
                    context_window['before'] = parent_chunks[start_idx:position]
                    
                    # Get next chunks (up to window_size)
                    end_idx = min(len(parent_chunks), position + window_size + 1)
                    context_window['after'] = parent_chunks[position+1:end_idx]
            
            enriched_result = {
                'result': result,
                'context_window': context_window
            }
            
            enriched_results.append(enriched_result)
        
        results_return = []
        for result in enriched_results:
            results_return.append(result['result'])
            for chunk in result['context_window']['before']:
                results_return.append(chunk)
            for chunk in result['context_window']['after']:
                results_return.append(chunk)

        return results_return

    def perform_hybrid_search(self,
                             query_text: str,
                             query_vector: List[float],
                             vector_fields: str = None,
                             search_options: Dict[str, Any] = None,
                             use_semantic_search: bool = False,
                             top: int = 10,
                             semantic_config_name: str = "default-semantic-config") -> List[Dict[str, Any]]:
        """
        Perform a hybrid search combining traditional keyword search with vector search.
        Args:
            query_text: The search query text
            vector_fields: List of fields to perform vector search on (default: ["text_vector"])
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
            "top": top,
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

class AIService:
    cognitive_client: CognitiveServicesManagementClient
    azure_account: azcsm.Account
    resource_group: ResourceGroup
    
    
    def __init__(self, 
                 resource_group: ResourceGroup,
                 cognitive_client: CognitiveServicesManagementClient,
                 azure_Account: azcsm.Account):
        self.resource_group = resource_group
        self.cognitive_client = cognitive_client
        self.azure_account = azure_Account

    def get_OpenAIClient(self, api_version:str) -> "OpenAIClient" :
        keys = self.cognitive_client.accounts.list_keys(self.resource_group.get_name(), self.azure_account.name)
        openai_client = AzureOpenAI(
            api_key=keys.key1,
            api_version=api_version,
            azure_endpoint= f"https://{self.azure_account.name}.openai.azure.com/",
        )
        return OpenAIClient(self, openai_client) 

    def get_models(self, azure_location: str = None) -> List[azcsm.Model]:
        if (azure_location is None): 
            azure_location = self.resource_group.azure_resource_group.location
        models_list = self.cognitive_client.models.list(azure_location)
        models = [model for model in models_list]
        return models 

    @staticmethod
    def get_model_details(model: azcsm.Model) -> Dict:
        """
        Get details for a specific model.
        
        Args:
            model_id: The model to be processed
            
        Returns:
            Dictionary with model details
        """
        try:
            info =  {
                "kind": model.kind,
                "name": model.model.name,
                "format": model.model.format,
                "version": model.model.version,
                "sju_name": model.sku_name,
            }
            return info
        except Exception as e:
            print(f"Error getting model '{model.model.name}': {str(e)}")
            return {}

    def get_deployments(self) -> List[azcsm.Deployment]:
        try:
            deployments = list(self.cognitive_client.deployments.list(self.resource_group.get_name(), self.azure_account.name))
            
            result = [ deployment for deployment in deployments ]
            return result
        except Exception as e:
            print(f"Error listing deployments: {str(e)}")
            return []
        
    def get_deployment(self, deployment_name:str) -> azcsm.Deployment : 
            deployment = self.cognitive_client.deployments.get( resource_group_name=self.resource_group.get_name(), account_name=self.azure_account.name, deployment_name=deployment_name )        
            return deployment

    @staticmethod        
    def get_deployment_details(deployment: azcsm.Deployment) -> Dict:
        """
        Get details for a specific deployment.
        
        Args:
            deployment: The deployment
            
        Returns:
            Dictionary with deployment details
        """
        try:
            # Handle missing properties safely
            deployment_info = {
                "name": deployment.name if hasattr(deployment, 'name') else "name not found",
                "status": "unknown"
            }
            # Add properties only if they exist
            if hasattr(deployment, 'properties'):
                props = deployment.properties
                if hasattr(props, 'model'):
                    deployment_info["model"] = props.model
                if hasattr(props, 'provisioning_state'):
                    deployment_info["status"] = props.provisioning_state
                # Handle scale settings if they exist
                if hasattr(props, 'scale_settings') and props.scale_settings is not None:
                    scale_settings = {}
                    if hasattr(props.scale_settings, 'scale_type'):
                        scale_settings["scale_type"] = props.scale_settings.scale_type
                    if hasattr(props.scale_settings, 'capacity'):
                        scale_settings["capacity"] = props.scale_settings.capacity
                    deployment_info["scale_settings"] = scale_settings
                # Add timestamps if they exist
                if hasattr(props, 'created_at'):
                    deployment_info["created_at"] = props.created_at
                if hasattr(props, 'last_modified'):
                    deployment_info["last_modified"] = props.last_modified
            return deployment_info
            
        except Exception as e:
            print(f"Error getting deployment '{deployment.name}': {str(e)}")
            return {}

    def create_deployment(self, 
                         deployment_name: str, 
                         model_name: str, 
                         model_version:str = None,
                         sku_name: str = "Standard",
                         capacity: int = 1) -> azcsm.Deployment:
        """
        Create a new model deployment in Azure OpenAI.
        
        Args:
            deployment_name: Name for the new deployment
            model: Base model name (e.g., 'gpt-4', 'text-embedding-ada-002')
            capacity: Number of tokens per minute in millions (TPM)
            scale_type: Scaling type (Standard, Manual)
            
        Returns:
            the Deployment when prepared
        """

        try:
            if model_version:
                model = azcsm.Model(name=model_name, version=model_version)
            else:
                model = azcsm.Model(name=model_name)

            deployment_properties = azcsm.DeploymentProperties(model=model)
            
            # Create SKU configuration
            sku = azcsm.Sku(name=sku_name, capacity=capacity)

            # properties = azcsm.DeploymentProperties(

            #     model=model,
            #     scale_settings=azcsm.DeploymentScaleSettings(
            #         scale_type=scale_type,
            #         capacity=capacity
            #     ),
            #     rai_policy_name="Microsoft.Default"
            # )
            poller = self.cognitive_client.deployments.begin_create_or_update(
                resource_group_name=self.resource_group.get_name(),
                account_name=self.azure_account.name,
                deployment_name=deployment_name,
                deployment=None, 
                parameteres = { 
                    "properties": deployment_properties,
                    "sku": sku
                }
            )
            deployment: azcsm.Deployment = poller.result()
            return deployment
            
        except Exception as e:
            print(f"Error creating deployment '{deployment_name}': {str(e)}")
            return {"error": str(e)}

    def delete_deployment(self, deployment_name: str) -> bool:
        """
        Delete a model deployment in Azure OpenAI.
        
        Args:
            deployment_name: Name of the deployment to delete
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Start the delete operation
            poller = self.cognitive_client.deployments.begin_delete(
                resource_group_name=self.resource_group.get_name(),
                account_name=self.azure_account.name,
                deployment_name=deployment_name
            )
            
            # Wait for the operation to complete
            result = poller.result()

            print(f"Successfully deleted deployment '{deployment_name}'")
            return True
        except Exception as e:
            print(f"Error deleting deployment '{deployment_name}': {str(e)}")
            return False

    def update_deployment(self, 
                         deployment_name: str, 
                         sku_name: str = "Standard",
                         capacity: int = 1) -> azcsm.Deployment:
        """
        Update an existing model deployment in Azure OpenAI.
        
        Args:
            deployment_name: Name of the deployment to update
            capacity: New capacity value (optional)
            scale_type: New scale type (optional)
            
        Returns:
            Dictionary with deployment details
        """
        try:
            deployment = self.get_deployment(deployment_name)

            model_props = deployment.properties.model
            model = azcsm.Model(
                # If the model is stored as a complex object, preserve its name/version
                name=model_props.name if hasattr(model_props, 'name') else model_props,
                version=model_props.version if hasattr(model_props, 'version') else None
            )            
            updated_sku = azcsm.Sku(name=sku_name, capacity=capacity)

            deployment_properties = azcsm.DeploymentProperties(model=model)

            poller = self.cognitive_client.deployments.begin_create_or_update(
                self.resource_group.get_name(),
                self.azure_account.name,
                deployment_name,
                parameters={
                    "properties": deployment_properties,
                    "sku": updated_sku
                }
            )            
            deployment = poller.result()
            return deployment
        except Exception as e:
            print(f"Error updating deployment '{deployment_name}': {str(e)}")
            return {"error": str(e)}

class OpenAIClient:
    ai_service: AIService 
    openai_client: AzureOpenAI

    def __init__(self, ai_service: AIService, openai_client: AzureOpenAI):
        self.ai_service = ai_service
        self.openai_client = openai_client
    
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def generate_embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Generate embeddings for text using Azure OpenAI.
        
        Args:
            text: The text to generate embeddings for
            model: The embedding model to use
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise e
    
    def generate_chat_completion(self, 
                                messages: List[Dict[str, str]], 
                                model: str, 
                                temperature: float = 0.7, 
                                max_tokens: int = 800) -> Dict[str, Any]:
        """
        Generate a chat completion using Azure OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The deployment name of the model
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Chat completion response
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            print(f"Error generating chat completion: {str(e)}")
            return {"error": str(e)}


    
##########################################################################################################################################################################
# Tests / Sample Use Cases 
##########################################################################################################################################################################
from config import(
    AZURE_TENANT_ID,
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_CONTAINER_NAME,

    AZURE_SEARCH_SERVICE_NAME,
    AZURE_INDEX_NAME,

    AZURE_OPENAI_SERVICE_NAME,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
)
def get_cognitive_sevices(): 
    identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
    subscription = identity.get_subscription(subscription_id=AZURE_SUBSCRIPTION_ID)
    cognitive_services = subscription.get_cognitive_client()
    resource_group = subscription.get_resource_group(AZURE_RESOURCE_GROUP)
    aiservice = resource_group.get_ai_service(AZURE_OPENAI_SERVICE_NAME)

    openaiclient = aiservice.get_OpenAIClient(AZURE_OPENAI_API_VERSION)
    embeddings = openaiclient.generate_embeddings("Hello, how are you?")
    
    models = aiservice.get_models()
    model_details = [ AIService.get_model_details(model) for model in models ]  

    deployments = aiservice.get_deployments() 
    deployment_details = [ AIService.get_deployment_details(deployment) for deployment in deployments ]

    # to be fixed 
    #aiservice.update_deployment("test-deployment", capacity=12)  
    #aiservice.delete_deployment("test-deployment")
    #deployment = aiservice.create_deployment( "test-deployment", "gpt-4o")

    return cognitive_services


def get_index(): 
    identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
    subscription = identity.get_subscription(subscription_id=AZURE_SUBSCRIPTION_ID)
    resource_group = subscription.get_resource_group(AZURE_RESOURCE_GROUP)
    search_service = subscription.get_search_service(AZURE_SEARCH_SERVICE_NAME)
    index =  search_service.get_index(AZURE_INDEX_NAME)
    return index

def update_index_schema():
    index = get_index()
    index.extend_index_schema([
        azsdim.SimpleField(name="readable_url", type=azsdim.SearchFieldDataType.String, filterable=True, facetable=True),
        azsdim.SimpleField(name="category", type=azsdim.SearchFieldDataType.String, filterable=True, facetable=True),
        #azsdim.SimpleField(name="publication_date", type=azsdim.SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        #azsdim.SearchableField(name="summary", type=azsdim.SearchFieldDataType.String, analyzer_name="el.lucene")
    ])
    return index

def update_fields():
    index = get_index()
    update_client = index.get_search_client()

    import urllib.parse

    def update_recs(recs: List[Dict[str, Any]]) -> int:
        for rec in recs:
            parsed = urllib.parse.unquote(rec["url"])
            rec["readable_url"] = parsed
            #rec["extension"] = rec["metadata_storage_file_extension"]
            #rec["content_type"] = rec["metadata_content_type"]

        result = update_client.upload_documents(documents=recs)
        succeeded = sum(1 for r in result if r.succeeded)
        return succeeded

    result = index.process_data_in_batches(index_name=None, transaction=update_recs, batch_size=100)
    return result

copy_index_name = AZURE_INDEX_NAME + "_2"

def get_index_copy(): 
    identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
    subscription = identity.get_subscription(subscription_id=AZURE_SUBSCRIPTION_ID)
    search_service = subscription.get_search_service(AZURE_SEARCH_SERVICE_NAME)
    index =  search_service.get_index(copy_index_name)
    return index

fields_to_copy = [
    "chunk_id",
    "parent_id",
    "chunk",
    "title",
    "text_vector",
    "url",
    "name",
    "readable_url",
    "extension",
    "content_type",
    "category",
]

def copy_index_structure():
    index = get_index()
    new_index = index.copy_index_structure(fields_to_copy=fields_to_copy, new_index_name=copy_index_name) 
def copy_index_data():
    index = get_index()
    result = index.copy_index_data(source_index_name=None, target_index_name=copy_index_name, fields_to_copy=fields_to_copy)
    return result

def copy_index():
    index = get_index()
    new_index_name = index.copy_index_structure(fields_to_copy=fields_to_copy, new_index_name=copy_index_name) 
    result = index.copy_index_data(source_index_name=None, target_index_name=copy_index_name, fields_to_copy=fields_to_copy)

def get_storage_account():
    identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
    subscription = identity.get_subscription(subscription_id=AZURE_SUBSCRIPTION_ID)
    resource_group = subscription.get_resource_group(AZURE_RESOURCE_GROUP)
    storage_account = resource_group.get_storage_account(AZURE_STORAGE_ACCOUNT_NAME)
    return storage_account

def get_containers():
    storage_account = get_storage_account()
    containers = storage_account.get_containers()
    return containers

def get_container():
    storage_account = get_storage_account()
    container = storage_account.get_container(AZURE_STORAGE_CONTAINER_NAME)
    return container

def get_blob_names():
    container = get_container()
    blob_names = container.get_blob_names()
    return blob_names    
    
if __name__ == "__main__":
    # Test AIService implementation
    from config import(
        AZURE_TENANT_ID,
        AZURE_CLIENT_ID,
        AZURE_CLIENT_SECRET,
        AZURE_SUBSCRIPTION_ID,
        AZURE_RESOURCE_GROUP,
        AZURE_AI_SERVICE_NAME,
        AZURE_OPENAI_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_EMBEDDING_MODEL,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_CHAT_MODEL
    )
    
    get_cognitive_sevices()
    exit() 

    # Test 1: Management operations with resource group
    print("\n=== Testing Management Operations ===")
    
    # Set up identity, subscription, and resource group
    identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
    subscription = identity.get_subscription(AZURE_SUBSCRIPTION_ID)
    resource_group = subscription.get_resource_group(AZURE_RESOURCE_GROUP)
    
    # Create a full-featured AI service client with auto-credential retrieval
    print("\nCreating AIService with auto-credential retrieval:")
    
    # Use alternate resource name if AZURE_AI_SERVICE_NAME is not available
    resource_name = AZURE_AI_SERVICE_NAME or "defaultaiservice"
    print(f"Using resource name: {resource_name}")
    
    management_service = AIService(
        resource_group=resource_group,
        resource_name=resource_name,
        api_version=AZURE_OPENAI_API_VERSION
    )
    
    # List available deployments
    print("\nListing deployments:")
    deployments = management_service.list_deployments()
    for deployment in deployments:
        print(f"- {deployment['name']} (model: {deployment['model']}, status: {deployment['status']})")
    
    # Get details for a specific deployment (if any exist)
    if deployments:
        first_deployment = deployments[0]['name']
        print(f"\nGetting details for deployment '{first_deployment}':")
        details = management_service.get_deployment_details(first_deployment)
        print(f"- Model: {details.get('model', 'N/A')}")
        print(f"- Status: {details.get('status', 'N/A')}")
        print(f"- Capacity: {details.get('scale_settings', {}).get('capacity', 'N/A')}")
        print(f"- Created: {details.get('created_at', 'N/A')}")
        print(f"- Last Modified: {details.get('last_modified', 'N/A')}")
    
    # Test 2: Direct OpenAI API client only (no management capabilities)
    print("\n=== Testing Direct OpenAI API Client (No Management) ===")
    
    print("\nCreating AIService with only OpenAI API credentials (no resource management):")
    api_only_service = AIService(
        api_key=AZURE_OPENAI_API_KEY,
        api_base=AZURE_OPENAI_ENDPOINT, 
        api_version=AZURE_OPENAI_API_VERSION
    )
    
    print("\nListing models with OpenAI API client:")
    models = api_only_service.list_available_models()
    for model in models:
        if 'id' in model and 'owned_by' in model:
            print(f"- {model['id']} (owned by: {model['owned_by']})")
        else:
            print(f"- {model}")
            
    print("\nGenerating embeddings with OpenAI API client...")
    test_embed_text = "Testing the embedding functionality with only OpenAI API credentials."
    embeddings = api_only_service.generate_embeddings(test_embed_text, AZURE_OPENAI_EMBEDDING_MODEL)
    print(f"Generated embedding vector with {len(embeddings)} dimensions")
    
    print("\nTests complete!")
    exit()



    query = "Ποιές είναι οι διαφορές στη διαδικασία έκδοσης χρεωστικών και πιστωτικών καρτών;"
    # query = "Ποιές είναι τα καταναλωτικά προιοντα που δίνει η Τράπεζα;"
    openai_client = get_embedding_client(AZURE_OPENAI_KEY, api_base=AZURE_OPENAI_ENDPOINT, api_version=azure_api_version )
    query_embedding = generate_embeddings(query, openai_client, model=azure_embedding_model)


    storage_accounts = subscription.get_storage_accounts()


    storage_account = resource_group.get_storage_account(storage_account_name)

    # create_result = resource_group.create_storage_account(storage_account_name + "a1", location)
    
    print("Getting or creating search service...")
    AZURE_SEARCH_SERVICE_NAME = "athenaindexes"
    search_service = subscription.get_search_service(AZURE_SEARCH_SERVICE_NAME)
    if search_service is None:
        print(f"Creating search service '{AZURE_SEARCH_SERVICE_NAME}'...")
        search_service = resource_group.create_search_service(AZURE_SEARCH_SERVICE_NAME, location)

    index =  search_service.get_index(AZURE_INDEX_NAME)
    
    semantic_config_name = "athena-index-full-semantic-configuration"
    if index is not None:
        result = index.perform_hybrid_search(query_text=query, 
                                    query_vector=query_embedding, 
                                    vector_fields="text_vector", 
                                    use_semantic_search=True,
                                    semantic_config_name=semantic_config_name) 
        
        
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
    print(f"Search service endpoint: {search_service.get_service_endpoint()}")
    
    # Define vector search configuration
    vector_search = get_std_vector_search()
    
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
    
    # Create the index with vector search configuration
    print(f"Creating or updating index '{AZURE_INDEX_NAME}'...")
    index = azsdim.SearchIndex(
        name=AZURE_INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )
    
    try:
        result = search_service.get_index_client().create_or_update_index(index)
        print(f"Index '{AZURE_INDEX_NAME}' created or updated successfully.")
        
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
        elif AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT and azure_openai_deployment:
            print("\nSetting up Azure OpenAI client...")
            search_service.setup_openai_client(
                api_key=AZURE_OPENAI_KEY,
                api_base=AZURE_OPENAI_ENDPOINT,
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