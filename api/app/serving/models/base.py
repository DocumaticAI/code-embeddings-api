import os
import time
from abc import abstractmethod
from pathlib import Path
from posixpath import split
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import uvicorn
import yaml
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from abc import abstractmethod, ABC


class AbstractTransformerEncoder(ABC): 
    '''
    class for the inheritance definitions for all of the encoders that will be usable as 
    partof the public embeddings API. 
    ''' 
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def allowed_languages(self):
        pass 

    @abstractmethod
    def load_model(self): 
        pass 

    @abstractmethod
    def make_embeddings(self):
        pass 
