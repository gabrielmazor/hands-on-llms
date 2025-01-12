import time
from typing import Any, Dict, List, Optional

import qdrant_client
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)
import os
import openai
import json
import atexit


from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.template import PromptTemplate


# This is a cache json file
# Define the cache file path in the user's home directory
CACHE_FILE = os.path.join(os.path.expanduser("~"), ".llm_cache.json")

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


# Save cache to disk at exit
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

# Register the save_cache function to be called at exit
cache = load_cache()
atexit.register(save_cache, cache)



def store_in_cache(prompt: str, response: str) -> None:
    """Store the response in the cache with the current date."""
    global cache

    # TODO : the time is here so ill be able to remove old entries from the cache at some point
    current_time = time.localtime()
    date_str = time.strftime("%Y-%m-%d", current_time)
    cache[prompt] = {"response": response, "date": date_str}

def fetch_from_cache(prompt: str) -> Optional[str]:
    """Fetch the response from the cache if it exists."""
    global cache
    if prompt in cache and cache[prompt].get("response"):
        return cache[prompt]["response"]
    return None

def call_openai_cached(engine, prompt, max_tokens, n, stop, temperature) -> str:
    """Call the OpenAI API with the given parameters and cache the response."""
    cached_response = fetch_from_cache(prompt)
    if cached_response:
        return cached_response

    response = openai.Completion.create(
        engine=engine, prompt=prompt, max_tokens=max_tokens, n=n, stop=stop, temperature=temperature
    )
    store_in_cache(prompt, response.choices[0].text)
    return response.choices[0].text

class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.

    Attributes:
    -----------
    top_k : int
        The number of top matches to retrieve from the vector store.
    embedding_model : EmbeddingModelSingleton
        The embedding model to use for encoding the question.
    vector_store : qdrant_client.QdrantClient
        The vector store to search for matches.
    vector_collection : str
        The name of the collection to search in the vector store.
    """

    top_k: int = 5
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]


    def handle_matched(self, matched: List[Dict[str, Any]], query: str) -> List[str]:
        """
        Handle the matched documents and return the context.
        choose from the correct field from the payload of the matched documents.
        handle empty summaries and duplicates.

        Parameters:
        -----------
        matched : List[Dict[str, Any]]
            The list of matched documents.

        query : str
            The input query sent by the user.

        Returns:
        --------
        List[str]
            The context from the matched documents.
        """


        # our options (brainstorming):
        # 1. remove empty summaries (maybe extend k in order to have more non enmpty summaries)
        # 2. using rule based approach to choose the best field to use
        # 3. summarize the missing summary from the text
        #    a. we might have to explain that its way cheaper than summarizing everything in the streaming pipeline
        #    b. in the real world, i would have a local cache in order to avoid summarising over and over again, or just upload the summary back to the db
         
        # try rule based. if not, use the model to summarize the text
        output = []
        for match in matched:
            if match.payload["summary"].strip():
                output.append(match.payload["summary"])

            
                # or the text if its less than 200 words
            elif len(match.payload["text"].split()) < 200:
                output.append(match.payload["text"])
            else:
                # else, summarize the text in the context of the query
                response = call_openai_cached(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=f"Can you summurise the relevant parts in this text in ~70 words and sort them by relevancy this query  : '{query}'\n\n" +
                        "\nOriginal Doc: f{d} ",
                    max_tokens=200,
                    n=1,
                    stop=None,
                    temperature=0.7
                )
                output.append(response)

        # replace not relevant with empty, some cleaning
        # this should be cleaned by the streaming pipeline
        output = [doc.replace("This article was generated by Benzinga's automated content engine and reviewed by an editor.", '') for doc in output]
        output = [doc.replace("generated by Benzinga's automated content engine and reviewed by an editor.", '') for doc in output]
        
        # remove duplicates and strip
        output = list(set(output))

        # strip and remove empty 
        output = [doc.strip() for doc in output if doc.strip()]
        

        return output

    def rank_documents(self, query: str, documents: List[str]) -> List[str]: 
        """
        Rank the documents using openAi API according to their relevant to the query.

        Parameters:
        -----------
        query : str
            The input query sent by the user.

        Returns:
        --------
        List[str]
            A list of ranked documents in decending order.
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        ranked_docs = []

        # remove empty
        documents = [doc for doc in documents if doc.strip()]
        # remove duplicates
        documents = list(set(documents))

        # replace line breaks with space
        documents = [doc.replace("\n", " ") for doc in documents]

        try:
            # ask the model to eliminate the irrelevant documents and rank the relevant ones
            # in reality, i might have used a local model for this task, as its cheaper
            # i ask the model to rank the documents based on the indices and not to return 
            # the text in order to reduce the number of toekens used
            response = call_openai_cached(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Hi, i am a financial analyst, can you generate me a list of indices " + 
            "(in this format [2,3,1,4]) of only the relevant docs to this query, sort them by relevancy  : '{query}'\n\n" +
                "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)]) +
                "\n\nRanked documents:",
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7
        )

            list_of_integers = json.loads(response)

            # remove duplicates
            list_of_integers = list(set(list_of_integers))
            # remove indexes that are out of range
            list_of_integers = [i for i in list_of_integers if (i <= len(documents) or i > 0)]

            # strip and return the ranked documents
            ranked_docs = [documents[doc_index-1].strip() for doc_index in list_of_integers]

             # leave a message to the LLM and mention that the context is ranked
            # remove this as it doesnt seem to improve
            #ranked_docs = ["This context is ranked in decending order:" ] + ranked_docs
        except Exception as e:
            # in case of hilusinations with bad indices or formats or that openai is down return an empty list
            pass
        return ranked_docs

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        _, quest_key = self.input_keys
        question_str = inputs[quest_key]

        cleaned_question = self.clean(question_str)
        # TODO: Instead of cutting the question at 'max_input_length', chunk the question in 'max_input_length' chunks,
        # pass them through the model and average the embeddings.
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)

        # TODO: Using the metadata, use the filter to take into consideration only the news from the last 24 hours
        # (or other time frame).
        matches = self.vector_store.search(
            query_vector=embeddings,
            k=self.top_k,
            collection_name=self.vector_collection,
        )


        # i query the model and ask for the relevant documents, but take their summary
        # what if there is no summary? i would use something else.
        enriched_documents = self.handle_matched(matches , question_str)


        _ranked_documents = self.rank_documents(question_str, enriched_documents)

        # in case i dont have the ranking. i will use the top k matches
        if not _ranked_documents:
            _ranked_documents = enriched_documents

            # remove duplicates
            _ranked_documents = list(set(_ranked_documents))

        # remove empty
        _ranked_documents = [doc.strip() for doc in _ranked_documents if doc.strip()]

        LIMIT_N_AFTER_RERANKING = 7
        context = "\n".join(_ranked_documents[:LIMIT_N_AFTER_RERANKING + 1]) + "\n" # the +1 is for the message to the LLM

        return {
            "context": context,
        }

    def clean(self, question: str) -> str:
        """
        Clean the input question by removing unwanted characters.

        Parameters:
        -----------
        question : str
            The input question to clean.

        Returns:
        --------
        str
            The cleaned question.
        """
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the chain"""

        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        """Returns a list of output keys for the chain"""

        return ["answer"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        inputs = self.clean(inputs)
        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"])
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                # TODO: Count tokens instead of using len().
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "prompt_template": self.template.infer_raw_template,
                    "usage.prompt_tokens": len(prompt["prompt"]),
                    "usage.total_tokens": len(prompt["prompt"]) + len(response),
                    "usage.actual_new_tokens": len(response),
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": response}

    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Cleans the inputs by removing extra whitespace and grouping broken paragraphs"""

        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs
