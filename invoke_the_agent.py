from pathlib import Path
from typing import List, Union, Generator, Iterator
import uuid

from pydantic import BaseModel

import os, time
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import (
    OpenAIPageableListOfThreadMessage, 
    ThreadMessage, 
    MessageContent,
    MessageTextUrlCitationAnnotation,
    MessageTextFileCitationAnnotation,
    MessageRole
)
from openai import AzureOpenAI


AZURE_CONNECTION_STRING= os.getenv("AZURE_CONNECTION_STRING")
AZURE_AGENT_ID = os.getenv("AZURE_AGENT_ID")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID =  os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

thread_id = "thread_dn1Fj6NzJe6dYTjJFYqoMQvk"

credential = ClientSecretCredential(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
project_client = AIProjectClient.from_connection_string( credential=credential, conn_str=AZURE_CONNECTION_STRING )

thread = project_client.agents.get_thread(thread_id=thread_id)

if thread is None:
    thread = project_client.agents.create_thread()
    thread_id = thread.id   

messages:OpenAIPageableListOfThreadMessage = project_client.agents.list_messages(thread_id=thread.id)

message: ThreadMessage = messages.get_last_message_by_role(MessageRole.AGENT)
content: List[MessageContent] = message.content
for content_item in content:
    if content_item.type == "text":
        initial_text = content_item.text["value"]
        text = content_item.text["value"]
        if "annotations" in content_item.text:
            annotations = content_item.text["annotations"]
            for annotation in annotations:
                if annotation.type == "url_citation":
                    annotation_text = annotation.text
                    url_citation = annotation.url_citation
                    if url_citation is not None:
                        annotation_title = url_citation.title
                        annotation_url = url_citation.url
                        text = text.replace(annotation_text, f"[{annotation_title}]({annotation_url})")
                elif annotation.type == "file_citation":
                    file_citation = annotation.file_citation
        if initial_text != text:
            content_item.text["value"] = text

        

annotations : List[MessageTextUrlCitationAnnotation] = message.url_citation_annotations


print(f"Messages: {messages}")
print ("=" * 100)

print(f"Last Message: {messages.data[0]}")
print ("=" * 50)
print(f"Last Message Content: {messages.data[0]["content"]}")

print ("=" * 50)

i = 0 
for msg_part in  messages.data[0]["content"]:
    
    print(f"Last Message part {i}: {msg_part["text"]}")
    text_value = msg_part["text"]["value"]
    annotations = msg_part["text"]["annotations"]



    #print(f"Last Message Text: {messages.data[0]["content"][0]["text"]["value"]}")


