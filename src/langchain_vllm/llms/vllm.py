from typing import Any, Dict, List, Optional, Annotated

import msgspec

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import pre_init
from pydantic import Field

from langchain_community.llms.openai import BaseOpenAI
from langchain_community.utils.openai import is_openai_v1

try:
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
except ImportError:
    pass


class VLLM(BaseLLM):
    """VLLM language model."""

    model: str = ""
    """The name or path of a HuggingFace Transformers model."""

    tensor_parallel_size: Optional[int] = 1
    """The number of GPUs to use for distributed execution with tensor parallelism."""

    trust_remote_code: Optional[bool] = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model 
    and tokenizer."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    best_of: Optional[int] = None
    """Number of output sequences that are generated from the prompt."""

    presence_penalty: float = 0.0
    """Float that penalizes new tokens based on whether they appear in the 
    generated text so far"""

    frequency_penalty: float = 0.0
    """Float that penalizes new tokens based on their frequency in the 
    generated text so far"""

    repetition_penalty: float = 1.0
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""

    temperature: float = 1.0
    """Float that controls the randomness of the sampling."""

    top_p: float = 1.0
    """Float that controls the cumulative probability of the top tokens to consider."""

    top_k: int = 0
    """Integer that controls the number of top tokens to consider."""

    min_p: float = 0.0
    """Represents the minimum probability for a token to be considered,
    relative to the probability of the most likely token. Must be in [0, 1].
    Set to 0 to disable this."""

    seed: int | None = None
    """Random seed to use for the generation."""

    use_beam_search: bool = False
    """Whether to use beam search instead of sampling."""

    stop: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated."""

    stop_token_ids: list[int] | None = None
    """Token IDs that stop the generation when they are generated. The returned
    output will contain the stop tokens unless the stop tokens are special
    tokens."""


    ignore_eos: bool = False
    """Whether to ignore the EOS token and continue generating tokens after 
    the EOS token is generated."""

    max_new_tokens: int = 512
    """Maximum number of tokens to generate per output sequence."""

    min_tokens: int = 0
    """Minimum number of tokens to generate per output sequence before EOS or
    `stop_token_ids` can be generated"""

    logprobs: Optional[int] = None
    """Number of log probabilities to return per output token."""

    prompt_logprobs: int | None = None
    """Number of log probabilities to return per prompt token.
    When set to -1, return all `vocab_size` log probabilities."""

    flat_logprobs: bool = False
    """Whether to return logprobs in flatten format (i.e. FlatLogprob)
    for better performance."""

    detokenize: bool = True
    """Whether to detokenize the output."""

    skip_special_tokens: bool = True
    """Whether to skip special tokens in the output."""

    spaces_between_special_tokens: bool = True
    """Whether to add spaces between special tokens in the output."""

    logits_processors: Any | None = None
    """Functions that modify logits based on previously generated tokens, and
    optionally prompt tokens as a first argument."""

    include_stop_str_in_output: bool = False
    """Whether to include the stop strings in output text."""

    truncate_prompt_tokens: Annotated[int, msgspec.Meta(ge=-1)] | None = None
    """If set to -1, will use the truncation size supported by the model. If
    set to an integer k, will use only the last k tokens from the prompt
    (i.e., left truncation). If set to `None`, truncation is disabled."""

    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE

    skip_clone: bool = False
    """Internal flag indicating that this SamplingParams instance is safe to
    reuse without cloning. When True, clone() will return self without
    performing a deep copy. This should only be set when the params object
    is guaranteed to be dedicated to a single request and won't be modified
    in ways that would affect other uses."""

    output_text_buffer_length: int = 0
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    structured_outputs: StructuredOutputsParams | None = None
    """Parameters for configuring structured outputs."""
    logit_bias: dict[int, float] | None = None
    """If provided, the engine will construct a logits processor that applies
    these logit biases."""
    allowed_token_ids: list[int] | None = None
    """If provided, the engine will construct a logits processor which only
    retains scores for the given token ids."""
    extra_args: dict[str, Any] | None = None
    """Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc. Not used by any in-tree sampling
    implementations."""

    bad_words: list[str] | None = None
    """Words that are not allowed to be generated. More precisely, only the
    last token of a corresponding token sequence is not allowed when the next
    generated token can complete the sequence."""
    _bad_words_token_ids: list[list[int]] | None = None

    skip_reading_prefix_cache: bool | None = None

    dtype: str = "auto"
    """The data type for the model weights and activations."""

    download_dir: Optional[str] = None
    """Directory to download and load the weights. (Default to the default 
    cache dir of huggingface)"""

    vllm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `vllm.LLM` call not explicitly specified."""

    client: Any = None  #: :meta private:

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            from vllm import LLM as VLLModel
        except ImportError:
            raise ImportError(
                "Could not import vllm python package. "
                "Please install it with `pip install vllm`."
            )

        values["client"] = VLLModel(
            model=values["model"],
            tensor_parallel_size=values["tensor_parallel_size"],
            trust_remote_code=values["trust_remote_code"],
            dtype=values["dtype"],
            download_dir=values["download_dir"],
            **values["vllm_kwargs"],
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vllm."""
        return {
            "n": self.n,
            "max_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "min_p": self.min_p,
            "prompt_logprobs": self.prompt_logprobs,
            "flat_logprobs": self.flat_logprobs,
            "stop_token_ids": self.stop_token_ids,
            "min_tokens": self.min_tokens,
            "stop": self.stop,
            "ignore_eos": self.ignore_eos,
            "use_beam_search": self.use_beam_search,
            "logprobs": self.logprobs,
        }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        from vllm import SamplingParams

        lora_request = kwargs.pop("lora_request", None)

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}

        # filter params for SamplingParams
        known_keys = SamplingParams.__annotations__.keys()
        sample_params = SamplingParams(
            **{k: v for k, v in params.items() if k in known_keys}
        )

        # call the model
        if lora_request:
            outputs = self.client.generate(
                prompts, sample_params, lora_request=lora_request
            )
        else:
            outputs = self.client.generate(prompts, sample_params)

        generations = []
        for output in outputs:
            text = output.outputs[0].text
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm"


class VLLMOpenAI(BaseOpenAI):
    """vLLM OpenAI-compatible API client"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""

        params: Dict[str, Any] = {
            "model": self.model_name,
            **self._default_params,
            "logit_bias": None,
        }
        if not is_openai_v1():
            params.update(
                {
                    "api_key": self.openai_api_key,
                    "api_base": self.openai_api_base,
                }
            )

        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm-openai"
