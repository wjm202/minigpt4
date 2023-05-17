import paddle
import argparse
import time
from PIL import Image
import warnings
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
# from paddlenlp.transformers import StoppingCriteria, StoppingCriteriaList
import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from minigpt4.common.registry import registry
from typing import Optional
import functools
import re
import types
from abc import ABC
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs:
            Additional stopping criteria specific kwargs.

    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.
"""

class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None
class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
    """

    def __init__(self, max_length: int):
        self.max_length = max_length

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length
class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = '###'
    sep2: str = None
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(system=self.system, roles=self.roles, messages=
            [[x, y] for x, y in self.messages], offset=self.offset,
            sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, conv_id
            =self.conv_id)

    def dict(self):
        return {'system': self.system, 'roles': self.roles, 'messages':
            self.messages, 'offset': self.offset, 'sep': self.sep, 'sep2':
            self.sep2, 'conv_id': self.conv_id}


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        for stop in self.stops:
            if paddle.all(x=stop == input_ids[0][-len(stop):].astype(dtype=
                'bool')).item():
                return True
        return False


CONV_VISION = Conversation(system=
    'Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.'
    , roles=('Human', 'Assistant'), messages=[], offset=2, sep_style=
    SeparatorStyle.SINGLE, sep='###')


class Chat:

    def __init__(self, model, vis_processor, device='gpu:0'):
        self.place = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [paddle.to_tensor(data=[835],place=self.place ), paddle.
            to_tensor(data=[2277, 29937],place=self.place)]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0
            ] and conv.messages[-1][1][-6:] == '</Img>':
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1,
        min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1,
        temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                'Warning: The number of tokens in current conversation exceeds the max length. The model will not see the contexts outside the range.'
                )
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        outputs = self.model.llama_model.generate(inputs_embeds=embs,
            max_new_tokens=max_new_tokens, stopping_criteria=self.
            stopping_criteria, num_beams=num_beams, do_sample=True,
            min_length=min_length, top_p=top_p, repetition_penalty=
            repetition_penalty, length_penalty=length_penalty, temperature=
            temperature)
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token,
            add_special_tokens=False)
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0)
        elif isinstance(image, paddle.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [self.model.llama_tokenizer(seg, return_tensors='pt',
            add_special_tokens=i == 0).input_ids for i, seg in
            enumerate(prompt_segs)]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for
            seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in
            pair] + [seg_embs[-1]]
        mixed_embs = paddle.concat(x=mixed_embs, axis=1)
        return mixed_embs












