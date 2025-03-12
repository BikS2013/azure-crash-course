from pathlib import Path
from typing import List, Union, Generator, Iterator
import uuid

from pydantic import BaseModel

import os, time
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import CodeInterpreterTool
from openai import AzureOpenAI


class Pipeline:
    class Valves(BaseModel):
        """Options to change from the WebUI"""
        AZURE_CONNECTION_STRING: str = ""
        AZURE_AGENT_ID: str = ""
        TENANT_ID: str = ""
        CLIENT_ID: str = ""
        CLIENT_SECRET: str = ""

    def __init__(self):
        self.threads = {}
        self.azure_project_connection_string = "swedencentral.api.azureml.ms;bbe2468f-27c5-4280-8647-e167a0a95125;autocode;autocode-project"
        self.azure_agent_id = "asst_rLZ7Eaz46HqvyvRfLnaKXoG1"
        self.name = "Azure Agent SearchAI"
        self.valves = self.Valves(**{
            "AZURE_CONNECTION_STRING": os.getenv("AZURE_CONNECTION_STRING",
                                                 "swedencentral.api.azureml.ms;bbe2468f-27c5-4280-8647-e167a0a95125;autocode;autocode-project"),
            "AZURE_AGENT_ID": os.getenv("AZURE_AGENT_ID", "asst_rLZ7Eaz46HqvyvRfLnaKXoG1"),
            "TENANT_ID": os.getenv("AZURE_TENANT_ID", ""),
            "CLIENT_ID": os.getenv("AZURE_CLIENT_ID", ""),
            "CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET", "")

        })
        
        if ( not self.valves.TENANT_ID or not self.valves.CLIENT_ID or not self.valves.CLIENT_SECRET):
            self.credential = None
            self.project_client = None
        else:
            self.credential = ClientSecretCredential(self.valves.TENANT_ID, self.valves.CLIENT_ID, self.valves.CLIENT_SECRET)
            self.project_client = AIProjectClient.from_connection_string(
                credential=self.credential, conn_str=self.valves.AZURE_CONNECTION_STRING
            )

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[Iterator[str], str]:
        

        print(f"pipe:{__name__} running")
        print("BODY: ", body)

        if 'task' in body['metadata']:
            return ""

        chat_id = body['user']['id']

        if not self.project_client:
            self.credential = ClientSecretCredential(self.valves.TENANT_ID, self.valves.CLIENT_ID, self.valves.CLIENT_SECRET)
            self.project_client = AIProjectClient.from_connection_string(
                credential=self.credential, conn_str=self.valves.AZURE_CONNECTION_STRING
            )

        if chat_id not in self.threads:


            # Create a thread
            thread = self.project_client.agents.create_thread()
            self.threads[chat_id] = thread.id
            print(f"Created thread, thread ID: {thread.id}")

            # Create a message
            message = self.project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=user_message,
            )
            print(f"Created message, message ID: {message.id}")

            # Run the agent
            run = self.project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=self.valves.AZURE_AGENT_ID)
            print(f"Run finished with status: {run.status}")

            if run.status == "failed":
                # Check if you got "Rate limit is exceeded.", then you want to get more quota
                print(f"Run failed: {run.last_error}")

            # Get messages from the thread
            messages = self.project_client.agents.list_messages(thread_id=thread.id)
            print(f"Messages: {messages}")

            # Get the last message from the sender
            last_msg = messages.get_last_text_message_by_role("assistant")
            if last_msg:
                print(f"Last Message: {last_msg.text.value}")

            # Print the file path(s) from the messages
            for file_path_annotation in messages.file_path_annotations:
                print(f"File Paths:")
                print(f"Type: {file_path_annotation.type}")
                print(f"Text: {file_path_annotation.text}")
                print(f"File ID: {file_path_annotation.file_path.file_id}")
                print(f"Start Index: {file_path_annotation.start_index}")
                print(f"End Index: {file_path_annotation.end_index}")
                self.project_client.agents.save_file(file_id=file_path_annotation.file_path.file_id,
                                                file_name=Path(file_path_annotation.text).name)

            return last_msg.text.value

        else:

            # Create a thread
            thread = self.project_client.agents.get_thread(self.threads[chat_id])

            if user_message == "Delete":
                print("Deleting thread")
                self.project_client.agents.delete_thread(self.threads[chat_id])
                self.threads.pop(chat_id)
                return "Thread deleted"

            print(f"Using existing thread, thread ID: {thread.id}")

            # Create a message
            message = self.project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=user_message,
            )
            print(f"Created message, message ID: {message.id}")

            # Run the agent
            run = self.project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=self.valves.AZURE_AGENT_ID)
            print(f"Run finished with status: {run.status}")

            if run.status == "failed":
                # Check if you got "Rate limit is exceeded.", then you want to get more quota
                print(f"Run failed: {run.last_error}")

            # Get messages from the thread
            messages = self.project_client.agents.list_messages(thread_id=thread.id)
            print(f"Messages: {messages}")

            # Get the last message from the sender
            last_msg = messages.get_last_text_message_by_role("assistant")
            if last_msg:
                print(f"Last Message: {last_msg.text.value}")

            # Print the file path(s) from the messages
            for file_path_annotation in messages.file_path_annotations:
                print(f"File Paths:")
                print(f"Type: {file_path_annotation.type}")
                print(f"Text: {file_path_annotation.text}")
                print(f"File ID: {file_path_annotation.file_path.file_id}")
                print(f"Start Index: {file_path_annotation.start_index}")
                print(f"End Index: {file_path_annotation.end_index}")
                self.project_client.agents.save_file(file_id=file_path_annotation.file_path.file_id,
                                                file_name=Path(file_path_annotation.text).name)

            return last_msg.text.value