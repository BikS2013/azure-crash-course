from logger import log, INFO, ERROR, WARNING
from search_endpoint import SearchEndPoint
from typing import Any, Dict, List

class SearchIndex:
    vector_store: SearchEndPoint
    AZURE_SEARCH_INDEX_NAME: str = None 
    metadata_fields: List[Dict[str, Any]] = None

    index_schema = {
        "name": AZURE_SEARCH_INDEX_NAME,
        "fields": [
            # fields for metadata e.g.
            # {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
            # etc.

            # Vector field for embeddings
            {
                "name": "content_vector",
                "type": "Collection(Edm.Single)",
                "dimensions": 1536,  # OpenAI embedding dimensions
                "vectorSearchProfile": "my-vector-config"
            }
        ],
        "vectorSearch": {
            "algorithms": [{
                "name": "my-hnsw", #hierarchical navigable small world
                "kind": "hnsw",
                # more information on the parameters can be found in
                # https://learn.microsoft.com/en-us/dotnet/api/azure.search.documents.indexes.models.hnswparameters?view=azure-dotnet
                "hnswParameters": {
                    "m": 10,                    # Recommended for ML embeddings
                    "efConstruction": 600,      # Higher for better accuracy
                    "efSearch": 600,            # Runtime search quality
                    "metric": "cosine"
                }
            }],
            "profiles": [{
                "name": "my-vector-config",
                "algorithm": "my-hnsw"
            }]
        }
    }

    def __init__(self, vector_store: SearchEndPoint, index_name: str, metadata_fields: List[Dict[str, Any]]):
        self.vector_store = vector_store
        self.AZURE_SEARCH_INDEX_NAME = index_name
        self.index_schema["name"] = index_name
        
        self.index_schema["fields"][0]['dimensions'] = len(self.vector_store.embeddings.embed_query('test'))

        self.metadata_fields = metadata_fields
        # Add any additional metadata fields
        for field in metadata_fields:
            if field not in self.index_schema["fields"]:
                self.index_schema["fields"].append(field)        
    
    def set_metadata_fields(self, value):
        self.metadata_fields = value
        existing_fields = [sc['name'] for sc in self.index_schema["fields"]]
        for field in self.metadata_fields:
            if field['name'] not in existing_fields: #self.index_schema["fields"]:
                self.index_schema["fields"].append(field)

    def ensure_metadata_fields(self) -> bool:
        """
        Ensure the index has all required fields, create or update if needed.
        """
        try:
            # Get current schema
            schema = self.vector_store.get_index_schema()
            if schema:
                # Check if all required fields exist with correct configuration
                current_fields = {f['name']: f for f in schema.get('fields', [])}
                missing_fields = []
                
                for required_field in self.metadata_fields:
                    field_name = required_field['name']
                    if field_name not in current_fields:
                        missing_fields.append(required_field)
                    # We could also check if existing fields have correct configuration
                    # but for now we'll just check existence
                
                if not missing_fields:
                    log(INFO, "Index has all required fields")
                    return True
                    
                log(WARNING, f"Missing fields in index: {[f['name'] for f in missing_fields]}")
                self.vector_store.delete_index()
                return self.vector_store.create_index_schema(self.index_schema)
            else:        
                return self.vector_store.update_index_schema(self.index_schema) 

        except Exception as e:
            log(ERROR, f"Error ensuring required fields: {str(e)}")
            return False
