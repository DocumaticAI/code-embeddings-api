from typing import Optional, Union

from pydantic import BaseModel


class ModelSchema(BaseModel):
    code_snippit: Optional[Union[list, str]]
    query: Optional[Union[list, str]]
    language: str
    task: Optional[str] = "embedding"
    response_max_len: Optional[int] = 64
    

"""
Example schemas for various tasks---

document generation-- 
{
  "function": "def foo(x): return True",
  "query": "documentation"
  "response_max_len: None
}

"""
