import azure_agent_pipeline as aap
from pathlib import Path
from typing import List, Union, Generator, Iterator
import uuid

from pydantic import BaseModel

import os, time
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient

class Pipeline(aap.Pipeline):
    def __init__(self):
        super().__init__()
        self.name = "Legal Cases Agent"
        
