import paddle
import logging
import random
from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from paddlenlp.transformers import LlamaForCausalLM
from paddlenlp.transformers import LlamaTokenizer


@registry.register_model('mini_gpt4')
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {'pretrain_vicuna':
        'configs/models/minigpt4.yaml'}

    def __init__(self, vit_model='eva_clip_g', q_former_model='/paddle/blip2_pretrained.pdparams',
        img_size=224, drop_path_rate=0, use_grad_checkpoint=False,
        vit_precision='fp16', freeze_vit=True, freeze_qformer=True,
        num_query_token=32, llama_model='', prompt_path='', prompt_template
        ='', max_txt_len=32, end_sym='\n', low_resource=False, device_8bit=0):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint,
            vit_precision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.stop_gradient = not False
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.stop_gradient = not False
            self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info('freeze vision encoder')
        print('Loading VIT Done')
        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token,
            self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.stop_gradient = not False
            self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.stop_gradient = not False
            logging.info('freeze Qformer')
        print('Loading Q-Former Done')
        print('Loading LLAMA')
        with paddle.amp.auto_cast():
            self.llama_tokenizer = LlamaTokenizer.from_pretrained("facebook/llama-7b")
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

            self.llama_model = LlamaForCausalLM.from_pretrained("facebook/vicuna-7b")
            for name, param in self.llama_model.named_parameters():
                param.stop_gradient = not False
        print('Loading LLAMA Done')
        self.llama_proj = paddle.nn.Linear(in_features=self.Qformer.config.
            hidden_size, out_features=self.llama_model.config.hidden_size)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if 
                '<ImageHere>' in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in
                filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list))
                )
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        if isinstance('cpu', paddle.dtype):
            dtype = 'cpu'
        elif isinstance('cpu', str) and 'cpu' not in ['cpu', 'gpu', 'ipu',
            'xpu']:
            dtype = 'cpu'
        elif isinstance('cpu', paddle.Tensor):
            dtype = 'cpu'.dtype
        else:
            dtype = self.ln_vision.dtype
        self.ln_vision.cast(dtype)
        self.ln_vision.astype(dtype='float32')
        if isinstance('cpu', paddle.dtype):
            dtype = 'cpu'
        elif isinstance('cpu', str) and 'cpu' not in ['cpu', 'gpu', 'ipu',
            'xpu']:
            dtype = 'cpu'
        elif isinstance('cpu', paddle.Tensor):
            dtype = 'cpu'.dtype
        else:
            dtype = self.visual_encoder.dtype
        self.visual_encoder.cast(dtype)
        self.visual_encoder.astype(dtype='float32')

    def encode_img(self, image):
        device = image.place
        if self.low_resource:
            self.vit_to_cpu()
            if isinstance('cpu', paddle.dtype):
                dtype = 'cpu'
            elif isinstance('cpu', str) and 'cpu' not in ['cpu', 'gpu',
                'ipu', 'xpu']:
                dtype = 'cpu'
            elif isinstance('cpu', paddle.Tensor):
                dtype = 'cpu'.dtype
            else:
                dtype = image.dtype
            image = image.cast(dtype)
        with self.maybe_autocast():
            if isinstance(device, paddle.dtype):
                dtype = device
            elif isinstance(device, str) and device not in ['cpu', 'gpu',
                'ipu', 'xpu']:
                dtype = device
            elif isinstance(device, paddle.Tensor):
                dtype = device.dtype
            else:
                dtype = self.ln_vision(self.visual_encoder(image)).dtype
            image_embeds = self.ln_vision(self.visual_encoder(image)).cast(
                dtype)
            if isinstance(device, paddle.dtype):
                dtype = device
            elif isinstance(device, str) and device not in ['cpu', 'gpu',
                'ipu', 'xpu']:
                dtype = device
            elif isinstance(device, paddle.Tensor):
                dtype = device.dtype
            else:
                dtype = paddle.ones(shape=image_embeds.shape[:-1], dtype=
                    'int64').dtype
            image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype=
                'int64').cast(dtype)
            query_tokens = self.query_tokens.expand(shape=[image_embeds.
                shape[0], -1, -1])
            query_output = self.Qformer.bert(query_embeds=query_tokens,
                encoder_hidden_states=image_embeds, encoder_attention_mask=
                image_atts, return_dict=True)
            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            if isinstance(image.place, paddle.dtype):
                dtype = image.place
            elif isinstance(image.place, str) and image.place not in ['cpu',
                 'gpu', 'ipu', 'xpu']:
                dtype = image.place
            elif isinstance(image.place, paddle.Tensor):
                dtype = image.place.dtype
            else:
                dtype = paddle.ones(shape=inputs_llama.shape[:-1], dtype=
                    'int64').dtype
            atts_llama = paddle.ones(shape=inputs_llama.shape[:-1], dtype=
                'int64').cast(dtype)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            p_before, p_after = prompt.split('<ImageHere>')
            if isinstance(img_embeds.place, paddle.dtype):
                dtype = img_embeds.place
            elif isinstance(img_embeds.place, str
                ) and img_embeds.place not in ['cpu', 'gpu', 'ipu', 'xpu']:
                dtype = img_embeds.place
            elif isinstance(img_embeds.place, paddle.Tensor):
                dtype = img_embeds.place.dtype
            else:
                dtype = self.llama_tokenizer(p_before, return_tensors='pt',
                    add_special_tokens=False).dtype
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors
                ='pt', add_special_tokens=False).cast(dtype)
            if isinstance(img_embeds.place, paddle.dtype):
                dtype = img_embeds.place
            elif isinstance(img_embeds.place, str
                ) and img_embeds.place not in ['cpu', 'gpu', 'ipu', 'xpu']:
                dtype = img_embeds.place
            elif isinstance(img_embeds.place, paddle.Tensor):
                dtype = img_embeds.place.dtype
            else:
                dtype = self.llama_tokenizer(p_after, return_tensors='pt',
                    add_special_tokens=False).dtype
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors=
                'pt', add_special_tokens=False).cast(dtype)
            p_before_embeds = self.llama_model.model.embed_tokens(
                p_before_tokens.input_ids).expand(shape=[batch_size, -1, -1])
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens
                .input_ids).expand(shape=[batch_size, -1, -1])
            wrapped_img_embeds = paddle.concat(x=[p_before_embeds,
                img_embeds, p_after_embeds], axis=1)
            wrapped_atts_img = atts_img[:, :1].expand(shape=[-1,
                wrapped_img_embeds.shape[1]])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        image = samples['image']
        img_embeds, atts_img = self.encode_img(image)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img,
                prompt)
        self.llama_tokenizer.padding_side = 'right'
        text = [(t + self.end_sym) for t in samples['text_input']]
        if isinstance(image.place, paddle.dtype):
            dtype = image.place
        elif isinstance(image.place, str) and image.place not in ['cpu',
            'gpu', 'ipu', 'xpu']:
            dtype = image.place
        elif isinstance(image.place, paddle.Tensor):
            dtype = image.place.dtype
        else:
            dtype = self.llama_tokenizer(text, return_tensors='pt', padding
                ='longest', truncation=True, max_length=self.max_txt_len,
                add_special_tokens=False).dtype
        to_regress_tokens = self.llama_tokenizer(text, return_tensors='pt',
            padding='longest', truncation=True, max_length=self.max_txt_len,
            add_special_tokens=False).cast(dtype)
        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens
            .input_ids == self.llama_tokenizer.pad_token_id, -100)
        if isinstance(image.place, paddle.dtype):
            dtype = image.place
        elif isinstance(image.place, str) and image.place not in ['cpu',
             'gpu', 'ipu', 'xpu']:
            dtype = image.place
        elif isinstance(image.place, paddle.Tensor):
            dtype = image.place.dtype
        else:
            dtype = paddle.ones(shape=[atts_img.shape[0], atts_img.shape[1] +
                1], dtype='int64').dtype
        empty_targets = paddle.ones(shape=[atts_img.shape[0], atts_img.
            shape[1] + 1], dtype='int64').cast(dtype).fill_(value=-100)
        targets = paddle.concat(x=[empty_targets, targets], axis=1)
        batch_size = img_embeds.shape[0]
        bos = paddle.ones(shape=[batch_size, 1], dtype=to_regress_tokens.
            input_ids.dtype) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        to_regress_embeds = self.llama_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = paddle.concat(x=[bos_embeds, img_embeds,
            to_regress_embeds], axis=1)
        attention_mask = paddle.concat(x=[atts_bos, atts_img,
            to_regress_tokens.attention_mask], axis=1)
        with self.maybe_autocast():
            outputs = self.llama_model(inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, return_dict=True, labels=targets
                )
        loss = outputs.loss
        return {'loss': loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get('vit_model', 'eva_clip_g')
        q_former_model = cfg.get('q_former_model',
            '/paddle/blip2_pretrained.pdparams'
            )
        img_size = cfg.get('image_size')
        num_query_token = cfg.get('num_query_token')
        llama_model = cfg.get('llama_model')
        drop_path_rate = cfg.get('drop_path_rate', 0)
        use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)
        vit_precision = cfg.get('vit_precision', 'fp16')
        freeze_vit = cfg.get('freeze_vit', True)
        freeze_qformer = cfg.get('freeze_qformer', True)
        low_resource = cfg.get('low_resource', False)
        device_8bit = cfg.get('device_8bit', 0)
        prompt_path = cfg.get('prompt_path', '')
        prompt_template = cfg.get('prompt_template', '')
        max_txt_len = cfg.get('max_txt_len', 32)
        end_sym = cfg.get('end_sym', '\n')
        model = cls(vit_model=vit_model, q_former_model=q_former_model,
            img_size=img_size, drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint, vit_precision=
            vit_precision, freeze_vit=freeze_vit, freeze_qformer=
            freeze_qformer, num_query_token=num_query_token, llama_model=
            llama_model, prompt_path=prompt_path, prompt_template=
            prompt_template, max_txt_len=max_txt_len, end_sym=end_sym,
            low_resource=low_resource, device_8bit=device_8bit)
        ckpt_path = cfg.get('ckpt', '')
        if ckpt_path:
            print('Load BLIP2-LLM Checkpoint: {}'.format(ckpt_path))
            ckpt = paddle.load(ckpt_path)
            msg = model.set_state_dict(state_dict=ckpt)
        return model