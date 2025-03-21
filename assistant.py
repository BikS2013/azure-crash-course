import os
from typing import Any, List, override
import openai as oai
import openai.types.beta as obeta
import openai.resources.beta as orbeta
from openai import AzureOpenAI, AssistantEventHandler
from openai.types.beta.vector_stores.vector_store_file_batch import VectorStoreFileBatch

class AIClient:
    azure_endpoint: str
    api_key: str
    api_version: str

    aiclient: AzureOpenAI

    def __init__(self, azure_endpoint, api_key, api_version):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.aiclient = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint)
    
    def get_ai_client(self):
        return self.aiclient
    def get_id(self):
        return self.aiclient.id

class VectorStore:
    name: str
    vector_store: obeta.VectorStore
    aiclient: AIClient

    def __init__(self, aiclient: AIClient, name: str, vector_store: obeta.VectorStore):
        self.aiclient = aiclient
        self.name = name
        self.vector_store = vector_store

    def get_id(self):
        return self.vector_store.id
    
    @staticmethod
    def create_vector_store(aiclient: AIClient, name: str):
        vector_store = aiclient.get_ai_client().beta.vector_stores.create(name=name)
        return VectorStore( aiclient, name, vector_store )
    
    @staticmethod
    def get_vector_store(aiclient: AIClient, name: str):
        vector_store = None
        vector_stores = list(aiclient.get_ai_client().beta.vector_stores.list())
        for item in vector_stores:
            if item.name == vector_store_name:
                vector_store = item
                return VectorStore( aiclient, name, vector_store )        
        return None
    
    def upload_files(self, file_paths: list) -> VectorStoreFileBatch:
        file_streams = [open(path, "rb") for path in file_paths]
        file_batch = self.aiclient.get_ai_client().beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=self.vector_store.id, files=file_streams
        )
        return file_batch

class FileSearchAssistant:
    name: str
    instructions: str
    model: str
    aiclient: AIClient

    def __init__(self, aiclient: AIClient, name: str, instructions: str, model: str, tools: list):
        self.aiclient = aiclient
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools
        self.assistant = None

    class EventHandler(AssistantEventHandler):
        client: AzureOpenAI

        def __init__(self, client:AIClient):
            super().__init__()
            self.client = client

        @override
        def on_text_created(self, text) -> None:
            print(f"\nassistant > ", end="", flush=True)

        @override
        def on_tool_call_created(self, tool_call):
            print(f"\nassistant > {tool_call.type}\n", flush=True)

        @override
        def on_message_done(self, message) -> None:
            # print a citation to the file searched
            message_content = message.content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{index}]"
                )
                if file_citation := getattr(annotation, "file_citation", None):
                    if ( self.client is None):
                        print("EventHandler: Missing Azure Open AI client ")
                    else:
                        cited_file = self.client.get_ai_client().files.retrieve(file_citation.file_id)
                        citations.append(f"[{index}] {cited_file.filename}")

            print(message_content.value)
            print("\n".join(citations))
    
    assistant: Any

    def get_beta(self) -> orbeta.Beta:
        return self.aiclient.get_ai_client().beta
    
    def get_assistant(self):
        self.get_beta().assistants.list()
        self.assistant = self.get_beta().assistants.create(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            tools=self.tools
        )
        return self.assistant
    
    vector_store_ids: List[str]
    def set_vector_stores(self, vector_store_ids: List[Any]):
        self.vector_store_ids = vector_store_ids
        if self.assistant is None:
            self.get_assistant()
        self.assistant = self.get_beta().assistants.update(
            assistant_id = self.assistant.id,
            tool_resources={"file_search": {"vector_store_ids": self.vector_store_ids}},
        )

    def get_thread(self, message: str) -> obeta.Thread:
        thread = self.get_beta().threads.create(
            messages=[
                {
                "role": "user",
                "content": message,
                # Attach the new file to the message.
                #   "attachments": [
                #     { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
                #   ],
                }
            ]
        )
        return thread



    def do_it(self, vector_store_ids: List[str] = None, thread:obeta.Thread = None, message :str = None   ):
        if self.assistant is None: 
            self.get_assistant()

        if vector_store_ids is not None:
            self.set_vector_stores(vector_store_ids)

        if thread is None:
            thread = self.get_thread(message)

        with self.get_beta().threads.runs.stream(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
            #instructions="Please address the user as Jane Doe. The user has a premium account.",
            event_handler= self.EventHandler(aiclient),
        ) as stream:
            stream.until_done()


assistant_system_prompt = """
You are an AI assistant that helps users learn from the information found in the source material.
Answer the query using only the sources you provided with.
Use bullets if the answer has multiple points.
Answer ONLY with the facts listed in the list of sources in your data. 
If there isn't enough information in the data, say you don't know.
Do not generate answers that don't use your data.
You must always answer in Greek and you must always cite your sources.
""" 


document_path = os.path.join(os.path.dirname(__file__), 'sample-documents')
print (document_path)
file_paths =  [] 

for root, dirs, files in os.walk(document_path):
    for file in files:
        file_paths.append(os.path.join(root, file))
        print(os.path.join(root, file))


file_paths = [path for path in file_paths if path.endswith(('.docx', '.pdf'))]
file_streams = [open(path, "rb") for path in file_paths]

api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
end_point = os.getenv("AZURE_OPENAI_ENDPOINT")
vector_store_name = "athena_documents2"
vector_store_name = "athena_documents"

aiclient = AIClient(azure_endpoint=end_point, api_key=api_key, api_version=api_version)
#vector_store = client.beta.vector_stores.create(name="athena_documents")

vector_store = VectorStore.get_vector_store(aiclient, vector_store_name)
#vector_store = VectorStore.create_vector_store(aiclient, vector_store_name)

# client = AzureOpenAI( api_key=api_key, api_version=api_version, azure_endpoint=end_point)

assistant = FileSearchAssistant(aiclient, 
                                name="AthenaAssistant", 
                                instructions=assistant_system_prompt, 
                                model="gpt-4o", 
                                tools=[{"type": "file_search"}])

vector_store_ids = [vector_store.vector_store.id]

assistant.set_vector_stores(vector_store_ids)
thread = assistant.get_thread("Ποιές είναι οι διαφορές στη διαδικασία έκδοσης χρεωστικών και πιστωτικών καρτών;")
assistant.do_it( thread=thread) 

# assistant.do_something()
#             # Create a thread and attach the file to the message
#             thread = aiclient.get_ai_client().beta.threads.create(
#                 messages=[
#                     {
#                     "role": "user",
#                     "content": "Ποιές είναι οι διαφορές στη διαδικασία έκδοσης χρεωστικών και πιστωτικών καρτών?",
#                     # Attach the new file to the message.
#                     #   "attachments": [
#                     #     { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
#                     #   ],
#                     }
#                 ]
#             )


exit()

if vector_store is not None:

    partial_files = file_paths[0:5]
    file_batch = vector_store.upload_files(partial_files)
    print(file_batch.status)
    print(file_batch.file_counts)

exit()

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
print(file_batch.status)
print(file_batch.file_counts)



exit() 
from openai import AzureOpenAI

class RAGAssistant: 
    azure_endpoint: str
    api_key: str
    api_version: str


    def __init__(self, azure_endpoint, api_key, api_version):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )


    def get_assistant(self, name: str, instructions: str,model: str):
        assistant = self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
            tools=[{"type": "file_search"}]
        )
        pass

    def get_vector_store(self, name: str):
        vector_store = self.client.beta.vector_stores.create(name=name)
        pass

############################################################################################################
# create assistant

import os
import json
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-08-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Create an assistant (enabling code interpreter tool)
assistant = client.beta.assistants.create(
    name="Data Visualization",
    instructions=f"You are a helpful AI assistant who makes interesting visualizations based on data." 
    f"You have access to a sandboxed environment for writing and testing code."
    f"When you are asked to create a visualization you should follow these steps:"
    f"1. Write the code."
    f"2. Anytime you write new code display a preview of the code to show your work."
    f"3. Run the code to confirm that it runs."
    f"4. If the code is successful display the visualization."
    f"5. If the code is unsuccessful display the error message and try to revise the code and rerun going through the steps from above again.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview" #You must replace this value with the deployment name for your model.
)



############################################################################################################
# upload files to the assistant

from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-05-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Create a vector store called "Financial Statements"
vector_store = client.beta.vector_stores.create(name="Financial Statements")
 
# Ready the files for upload to OpenAI
file_paths = ["mydirectory/myfile1.pdf", "mydirectory/myfile2.txt"]
file_streams = [open(path, "rb") for path in file_paths]
 
# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)
 
# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

############################################################################################################
# update assistant to use a new vector store

assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

############################################################################################################
# create a thread 

# Upload the user provided file to OpenAI
message_file = client.files.create(
  file=open("mydirectory/myfile.pdf", "rb"), purpose="assistants"
)
 
# Create a thread and attach the file to the message
thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": "How many company shares were outstanding last quarter?",
      # Attach the new file to the message.
      "attachments": [
        { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
      ],
    }
  ]
)
 
# The thread now has a vector store with that file in its tool resources.
print(thread.tool_resources.file_search)

############################################################################################################
# create a run and check the output 

from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
 
client = OpenAI()
 
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    @override
    def on_message_done(self, message) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        print(message_content.value)
        print("\n".join(citations))


# Then, we use the stream SDK helper
# with the EventHandler class to create the Run
# and stream the response.

with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
    event_handler=EventHandler(),
) as stream:
    stream.until_done()


