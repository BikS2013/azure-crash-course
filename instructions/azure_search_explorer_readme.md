# Azure Setup Guide for Mac: Exploring Azure AI Search Services

Here's a comprehensive guide to set up your Mac for Azure development and obtain credentials for exploring Azure AI Search services.

## 1. Install Required Tools

### Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Azure CLI
```bash
brew update && brew install azure-cli
```

### Install Python (if needed)
```bash
brew install python
```

### Install Required Python Packages
```bash
pip install azure-identity azure-search-documents azure-core
```

## 2. Authenticate with Azure

### Sign in to Azure CLI
```bash
az login
```
This will open a browser window for authentication.

### Verify your account details
```bash
az account show
```

## 3. Get Azure Credentials

### Option 1: Use Azure CLI Credential (Simplest)

Create a Python file (`azure_search_explorer.py`):

```python
from azure.identity import AzureCliCredential
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Use CLI-based authentication
credential = AzureCliCredential()

# Print subscription to verify
from azure.mgmt.resource import SubscriptionClient
subscription_client = SubscriptionClient(credential)
subscription = next(subscription_client.subscriptions.list())
print(f"Using subscription: {subscription.display_name} ({subscription.subscription_id})")

# Now you can use this credential for Azure services
```

### Option 2: Get Specific Subscription Details

If you need to work with a specific subscription:

```bash
# List all subscriptions
az account list --output table

# Set a specific subscription as default
az account set --subscription "Your-Subscription-Name-or-ID"
```

## 4. Exploring Azure AI Search Services

Add this to your Python code:

```python
from azure.mgmt.search import SearchManagementClient

# Get your subscription ID
subscription_id = next(subscription_client.subscriptions.list()).subscription_id

# Create a Search Management client
search_mgmt_client = SearchManagementClient(credential, subscription_id)

# List all search services in your subscription
print("Available Search Services:")
for service in search_mgmt_client.services.list_by_subscription():
    print(f"- {service.name} (Location: {service.location})")

# To connect to a specific service (you'll need the admin key)
service_name = "your-search-service-name"
resource_group = "your-resource-group"

# Get the admin key
admin_key = search_mgmt_client.admin_keys.get(
    resource_group_name=resource_group,
    search_service_name=service_name
).primary_key

# Connect to the service
endpoint = f"https://{service_name}.search.windows.net"
search_client = SearchClient(
    endpoint=endpoint,
    index_name="your-index-name",
    credential=AzureKeyCredential(admin_key)
)

# Now you can perform search operations
# For example, search for documents
results = search_client.search(search_text="your search query")
for result in results:
    print(result)
```

## 5. Create a Complete Explorer Script

## 6. Usage Instructions

1. **Save the script**: Save the above code to a file named `azure_search_explorer.py`

2. **Make it executable**:
   ```bash
   chmod +x azure_search_explorer.py
   ```

3. **Run the script**:
   ```bash
   ./azure_search_explorer.py
   ```

4. **Follow the interactive prompts**:
   - The script will authenticate you using Azure CLI
   - It will list available subscriptions for you to select
   - It will show all search services in that subscription
   - After selecting a service, it will show available indexes
   - You can then explore the index schema and search for documents

## 7. Troubleshooting

- **If authentication fails**: Ensure you've run `az login` first
- **If you don't see your search services**: Verify you're using the correct subscription
- **If you can't access indexes**: You might not have sufficient permissions. Ask your Azure administrator for access

## 8. Next Steps

After you've explored your search services, you might want to:

1. Learn about cognitive search capabilities
2. Create custom skillsets for AI enrichment
3. Explore semantic search features
4. Build a simple web interface to showcase search functionality

Would you like more information on any specific part of this guide?