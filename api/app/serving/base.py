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
    def load_tokenizer(self):
        pass

    @abstractmethod
    def load_model(self): 
        pass 

    @abstractmethod
    def make_embeddings(self):
        pass 



class UniXEncoderBase(nn.Module):
    def __init__(self, encoder):
        super(UniXEncoderBase, self).__init__()
        self.encoder = encoder
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(
                1
            ).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(
                1
            ).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


class UniXCoderEmbedder(object):
    """ """

    def __init__(self):

        self.config_path = Path(__file__).parent / "config.yaml"
        self.model_args = yaml.safe_load(self.config_path.read_text())
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_args["UniXCoder"]["base_model"]
        )
        self.config = RobertaConfig.from_pretrained(
            self.model_args["UniXCoder"]["base_model"]
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.serving_batch_size = 32
        self.model_languages = ["python", "javascript", "go", "java"]
        self.model_dict = {}
        for language in self.model_languages:
            self.model_dict[language] = self.load_model(model_language=language)

    @staticmethod
    def split_list_equal_chunks(list_object, split_length):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(list_object), split_length):
            yield list_object[i : i + split_length]

    def load_model(self, model_language: str):
        """
        Abstract loader for loading models from disk into embedding models for each language
        Arguments
        ---------
        model_language (str):
            a programming language for which to do search. Currently, each language has its own model
        Returns
        -------
        model_to_load (BaseEncoder):
            an instance of a wrapped roberta model that has been finetuned on the codesearchnet corpus
        """
        start = time.time()
        model = UniXEncoderBase(
            encoder=RobertaModel.from_pretrained(
                "microsoft/unixcoder-base"
            )
        )
        model_to_load = model.module if hasattr(model, "module") else model

        print(
            "Search retrieval model for language set {} loaded correctly to device {} in {} seconds".format(
                model_language, self.device, time.time() - start
            )
        )
        return model_to_load.to(self.device)

    def tokenize(
        self,
        inputs: Union[List[str], str],
        mode="<encoder-only>",
        max_length=256,
        padding=True,
    ) -> list:
        """
        Convert string to token ids
        Parameters:
        * `inputs`- list of input strings.
        * `max_length`- The maximum total source sequence length after tokenization.
        * `padding`- whether to pad source sequence length to max_length.
        * `mode`- which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
        """
        assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]

        tokenizer = self.tokenizer

        if isinstance(inputs, str):
            inputs = [inputs]

        tokens_ids = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            if mode == "<encoder-only>":
                tokens = tokens[: max_length - 4]
                tokens = (
                    [tokenizer.cls_token, mode, tokenizer.sep_token]
                    + tokens
                    + [tokenizer.sep_token]
                )
            elif mode == "<decoder-only>":
                tokens = tokens[-(max_length - 3) :]
                tokens = [tokenizer.cls_token, mode, tokenizer.sep_token] + tokens
            else:
                tokens = tokens[: max_length - 5]
                tokens = (
                    [tokenizer.cls_token, mode, tokenizer.sep_token]
                    + tokens
                    + [tokenizer.sep_token]
                )

            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            if padding:
                tokens_id = tokens_id + [self.config.pad_token_id] * (
                    max_length - len(tokens_id)
                )
            tokens_ids.append(tokens_id)
        return tokens_ids

    def make_inference_minibatch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        language: str,
        embedding_type: str,
    ) -> list:
        """
        Takes in a either a single string of a code or a query or a small batch, and returns an embedding for each input.
        Follows standard ML embedding workflow, tokenization, token tensor passed to model, embeddings
        converted to cpu and then turned to lists and returned, Most parameters are for logging.
        Parameters
        ----------
        string_batch - Union[list, str]:
            either a single example or a list of examples of a query or piece of source code to be embedded
        max_length_tokenizer - int:
            the max length for a snippit before it is cut short. 256 tokens for code, 128 for queries.
        language - str:
            logging parameter to display the programming language being inferred upon
        embedding_type - str:
            logging parameter to display the task for embedding, query or code.
        """
        start = time.time()
        model = self.model_dict[language]

        code_token_ids = self.tokenize(
            string_batch, max_length=max_length_tokenizer, mode="<encoder-only>"
        )
        code_source_ids = torch.tensor(code_token_ids).to(self.device)
        inference_embeddings = (
            model.forward(code_inputs=code_source_ids).cpu().detach().tolist()
        )
        if isinstance(string_batch, str):
            print(
                f"inference_logged-  batch_size:{1}, language:{language}, request_type:{embedding_type}, inference_time:{time.time()-start:.4f}, average_inference_time:{((time.time()-start)):.4f}"
            )
        elif isinstance(string_batch, list):
            print(
                f"inference_logged-  batch_size:{len(string_batch)}, language:{language}, request_type:{embedding_type}, inference_time:{time.time()-start:.4f}, average_inference_time:{((time.time()-start)/len(string_batch)):.4f}"
            )

        return inference_embeddings

    def make_inference_batch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        language: str,
        embedding_type: str,
    ) -> list:
        """
        Takes in a either a single string of a code or a query or a batch of any size, and returns an embedding for each input.
        Follows standard ML embedding workflow, tokenization, token tensor passed to model, embeddings
        converted to cpu and then turned to lists and returned, Most parameters are for logging.
        Parameters
        ----------
        string_batch - Union[list, str]:
            either a single example or a list of examples of a query or piece of source code to be embedded
        max_length_tokenizer - int:
            the max length for a snippit before it is cut short. 256 tokens for code, 128 for queries.
        language - str:
            logging parameter to display the programming language being inferred upon
        embedding_type - str:
            logging parameter to display the task for embedding, query or code.
        """
        start = time.time()
        model = self.model_dict[language]
        code_embeddings_list = []

        # Sort inputs by list
        split_code_batch = self.split_list_equal_chunks(
            string_batch, self.serving_batch_size
        )

        for minibatch in split_code_batch:
            code_token_ids = self.tokenize(
                minibatch, max_length=max_length_tokenizer, mode="<encoder-only>"
            )
            code_source_ids = torch.tensor(code_token_ids).to(self.device)
            code_embeddings_list.append(
                model.forward(code_inputs=code_source_ids).cpu().detach().tolist()
            )
            del code_source_ids
            torch.cuda.empty_cache()

        inference_embeddings = [x for xs in code_embeddings_list for x in xs]
        print(
            f"inference_logged-  batch_size:{len(string_batch)}, language:{language}, request_type:{embedding_type}, inference_time:{time.time()-start:.4f}, average_inference_time:{((time.time()-start)/len(string_batch)):.4f}"
        )
        return inference_embeddings

    def make_embeddings(
        self, code_batch: Union[list, str], query_batch: Union[list, str], language: str
    ) -> dict:
        """
        Wrapping function for making inference on batches of source code or queries to embed them.
        Takes in a single or batch example for code and queries along with a programming language to specify the
        language model to use, and returns a list of lists which corresponds to embeddings for each item in
        the batch.
        Parameters
        ----------
        code_batch - Union[list, str]:
            either a list or single example of a source code snippit to be embedded
        query_batch - Union[list, str]:
            either a list or single example of a query to be embedded to perform search
        language - str:
            a programming language that is required to specify the embedding model to use (each language that
            has been finetuned on has it's own model currently)
        """
        # Make embeddings for batches of code in request body
        if code_batch:
            if (
                isinstance(code_batch, list)
                and len(code_batch) > self.serving_batch_size
            ):
                code_embeddings = self.make_inference_batch(
                    string_batch=code_batch,
                    max_length_tokenizer=256,
                    language=language,
                    embedding_type="code",
                )
            else:
                code_embeddings = self.make_inference_minibatch(
                    string_batch=code_batch,
                    max_length_tokenizer=256,
                    language=language,
                    embedding_type="code",
                )
        else:
            code_embeddings = None

        # Make embeddings for batches of queries in request body
        if query_batch:
            if (
                isinstance(query_batch, list)
                and len(query_batch) > self.serving_batch_size
            ):
                query_embeddings = self.make_inference_batch(
                    string_batch=query_batch,
                    max_length_tokenizer=128,
                    language=language,
                    embedding_type="query",
                )
            else:
                query_embeddings = self.make_inference_minibatch(
                    string_batch=query_batch,
                    max_length_tokenizer=128,
                    language=language,
                    embedding_type="query",
                )
        else:
            query_embeddings = None

        return {
            "code_batch": {
                "code_strings": code_batch,
                "code_embeddings": code_embeddings,
            },
            "query_batch": {
                "query_strings": query_batch,
                "query_embeddings": query_embeddings,
            },
        }