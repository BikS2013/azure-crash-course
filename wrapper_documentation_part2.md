## Core Classes

### Authentication and Identity Management

#### Identity Class

The `Identity` class is the entry point for authentication with Azure services. It uses service principal authentication with a tenant ID, client ID, and client secret.

**Purpose**: Authenticate with Azure and provide access to subscriptions.

**Key Attributes**:
- `tenant_id`: Azure AD tenant ID
- `client_id`: Service principal client ID
- `client_secret`: Service principal client secret
- `credential`: Azure `ClientSecretCredential` object
- `subscription_client`: Azure `SubscriptionClient` for accessing subscriptions

**Key Methods**:
- `get_credential()`: Returns the Azure credential object
- `get_subscriptions()`: Lists all available subscriptions
- `get_subscription(subscription_id)`: Gets a specific subscription by ID

**Usage Example**:
```python
# Create an Identity object
identity = Identity(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET")
)

# Get available subscriptions
subscriptions = identity.get_subscriptions()
for sub in subscriptions:
    print(f"Subscription: {sub.display_name} ({sub.subscription_id})")

# Get a specific subscription
subscription = identity.get_subscription(os.getenv("AZURE_SUBSCRIPTION_ID"))
```

### Resource Management

#### Subscription Class

The `Subscription` class represents an Azure subscription and provides access to resource groups and services within that subscription.

**Purpose**: Manage resource groups and access Azure services within a subscription.

**Key Attributes**:
- `identity`: Reference to the Identity object
- `subscription`: Azure subscription object
- `subscription_id`: Azure subscription ID
- `resource_client`: Azure `ResourceManagementClient` for managing resources
- `storage_client`: Azure `StorageManagementClient` for managing storage accounts

**Key Methods**:
- `get_resource_group(group_name)`: Gets a resource group by name
- `create_resource_group(group_name, location)`: Creates a new resource group
- `get_search_services()`: Lists all search services in the subscription
- `get_search_service(service_name)`: Gets a specific search service by name
- `get_storage_accounts()`: Lists all storage accounts in the subscription
- `get_cognitive_client()`: Gets a client for managing cognitive services

**Usage Example**:
```python
# Get a resource group
resource_group = subscription.get_resource_group("my-resource-group")

# Create a resource group if it doesn't exist
if resource_group is None:
    resource_group = subscription.create_resource_group("my-resource-group", "eastus")

# List all search services
search_services = subscription.get_search_sevices()
for service in search_services:
    print(f"Search Service: {service.name}")

# Get a specific search service
search_service = subscription.get_search_service("my-search-service")
```

#### ResourceGroup Class

The `ResourceGroup` class represents an Azure resource group and provides methods to manage resources within that group.

**Purpose**: Manage Azure resources within a resource group.

**Key Attributes**:
- `subscription`: Reference to the Subscription object
- `azure_resource_group`: Azure resource group object

**Key Methods**:
- `get_name()`: Gets the name of the resource group
- `get_resources()`: Lists all resources in the group
- `create_search_service(name, location)`: Creates a new search service
- `get_storage_account(account_name)`: Gets a storage account by name
- `create_storage_account(account_name, location)`: Creates a new storage account
- `get_ai_service(service_name)`: Gets an AI service by name

**Usage Example**:
```python
# Get the resource group name
group_name = resource_group.get_name()

# List all resources in the group
resources = resource_group.get_resources()
for resource in resources:
    print(f"Resource: {resource.name} (Type: {resource.type})")

# Create a search service
search_service = resource_group.create_search_service("my-search-service", "eastus")

# Get or create a storage account
storage_account = resource_group.get_storage_account("mystorageaccount")
if storage_account is None:
    storage_account = resource_group.create_storage_account("mystorageaccount", "eastus")