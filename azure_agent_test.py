import os
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient



if __name__ == "__main__":
    tenant_id = os.getenv("AZURE_TENANT_ID")
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")


    credential = ClientSecretCredential(tenant_id, client_id, client_secret)

    con_str = "eastus2.api.azureml.ms;51dfc225-0c48-4bf4-bbcc-ce78272befc5;bikstestrg;biks-rag"

    project_client = AIProjectClient.from_connection_string(credential=credential, conn_str=con_str)

    thread = project_client.agents.create_thread()

    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="tell me a story",
    )

    agent_id = "asst_BbBpU1oBAvuZn48oslKzyvLH"
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent_id)

    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        # Check if you got "Rate limit is exceeded.", then you want to get more quota
        print(f"Run failed: {run.last_error}")

    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")




