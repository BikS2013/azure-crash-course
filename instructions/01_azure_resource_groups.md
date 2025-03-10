# Q
I need to give me a step-by-step guide on how I can get a list of the ResourceGroups I have access to in a given subscription, 
how to create a new ResourceGroup, 
how to get a list of the resources in a Resource Group

# A
I'll provide you with a step-by-step guide on working with Azure Resource Groups using the Azure CLI. Based on the Azure setup you've already completed according to the document, I'll assume you have Azure CLI installed and authenticated.

## How to List Resource Groups in a Subscription

1. First, ensure you're logged in:
   ```bash
   az login
   ```

2. To list all resource groups in your current subscription:
   ```bash
   az group list --output table
   ```

3. If you want to use a specific subscription:
   ```bash
   # First list available subscriptions
   az account list --output table
   
   # Set the subscription you want to use
   az account set --subscription "Your-Subscription-Name-or-ID"
   
   # Then list resource groups in that subscription
   az group list --output table
   ```

4. For more detailed output in JSON format:
   ```bash
   az group list --output json
   ```

## How to Create a New Resource Group

1. To create a new resource group:
   ```bash
   az group create --name "YourNewResourceGroup" --location "eastus"
   ```

2. Replace "YourNewResourceGroup" with your desired resource group name and "eastus" with your preferred Azure region. Some common regions are:
   - eastus (East US)
   - westus (West US)
   - westeurope (West Europe)
   - southeastasia (Southeast Asia)

3. You can verify the creation by listing your resource groups again:
   ```bash
   az group list --output table
   ```

## How to List Resources in a Resource Group

1. To list all resources in a specific resource group:
   ```bash
   az resource list --resource-group "YourResourceGroup" --output table
   ```

2. For more detailed information in JSON format:
   ```bash
   az resource list --resource-group "YourResourceGroup" --output json
   ```

3. To filter by resource type (for example, only Storage accounts):
   ```bash
   az resource list --resource-group "YourResourceGroup" --resource-type "Microsoft.Storage/storageAccounts" --output table
   ```

4. To get detailed information about a specific resource:
   ```bash
   az resource show --resource-group "YourResourceGroup" --name "YourResourceName" --resource-type "Microsoft.Storage/storageAccounts"
   ```

These commands should help you manage your Azure Resource Groups efficiently. Let me know if you need any clarification or have additional questions!