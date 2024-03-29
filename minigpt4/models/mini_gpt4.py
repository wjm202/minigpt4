
import paddle
import logging
import random
from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from minigpt4.models.eva_vit import VisionTransformer
from paddlenlp.transformers import LlamaTokenizer
from paddlenlp.ops import transfer_param
import os
os.environ["FLAGS_use_cuda_managed_memory"]="true"

@registry.register_model('mini_gpt4')
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {'pretrain_vicuna':
        'configs/models/minigpt4.yaml'}

    def __init__(self, vit_model='eva_clip_g', q_former_model='blip2_pretrained_flant5xxl.pdparams',
        img_size=224, drop_path_rate=0, use_grad_checkpoint=False,
        vit_precision='fp16', freeze_vit=True, freeze_qformer=True,
        num_query_token=32, llama_model='', prompt_path='', prompt_template='', max_txt_len=32, end_sym='\n', low_resource=False, device_8bit=0):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint,
            vit_precision)
        convert_weights_to_dtype(self.visual_encoder,dtype="float16")
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
            self.visual_encoder.num_features)#self.visual_encoder.num_features：1408
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.stop_gradient = not False
            self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.stop_gradient = not False
            logging.info('freeze Qformer')
        # convert_weights_to_dtype(self.Qformer,dtype="float16")
        self.query_tokens = transfer_param(self.query_tokens, restore_data=True, dtype="float32")
        print('Loading Q-Former Done')
        print('Loading LLAMA')

        self.llama_tokenizer = LlamaTokenizer.from_pretrained("facebook/llama-7b")
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_model = LlamaForCausalLM.from_pretrained("/paddle/MiniGPT-4/vicuna-7b",dtype="float32")
        convert_weights_to_dtype(self.llama_model,dtype="float16")
        for name, param in self.llama_model.named_parameters():
            param.stop_gradient = not False
        print('Loading LLAMA Done')
        self.llama_proj = paddle.nn.Linear(in_features=self.Qformer.config.hidden_size, out_features=self.llama_model.config.hidden_size)
        self.llama_proj.weight = transfer_param(self.llama_proj.weight,restore_data=True, dtype="float32")
        self.llama_proj.bias= transfer_param(self.llama_proj.bias,restore_data=True, dtype="float32")
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
        
    def encode_img(self, image):
        with self.maybe_autocast():
    
            image_embeds = self.ln_vision(self.visual_encoder(image.astype("float32")).astype("float32"))
            image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype=
                'int64')
            # import numpy as np
            # image_embeds=paddle.to_tensor(np.load('/paddle/MiniGPT-4/image_embeds.npy'))
            query_tokens = self.query_tokens.expand(shape=[image_embeds.
                shape[0], -1, -1]).astype("float32")
            # query_tokens=paddle.to_tensor(np.load('/paddle/MiniGPT-4/query_tokens.npy'))
            query_output = self.Qformer.bert(query_embeds=query_tokens.astype("float32"),
                encoder_hidden_states=image_embeds.astype("float32"), encoder_attention_mask=
                image_atts, return_dict=True)
            inputs_llama = self.llama_proj(query_output.last_hidden_state.astype("float32")).astype("float16")
            atts_llama = paddle.ones(shape=inputs_llama.shape[:-1], dtype=
                'int64')
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors
                ='pd', add_special_tokens=False)
            p_after_tokens = self.llama_tokenizer(p_after, return_tensors=
                'pd', add_special_tokens=False)
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
        # import numpy as np
        # image=paddle.to_tensor(np.load('/paddle/MiniGPT-4/image.npy')).astype("float32")
        samples= samples[0]
        image = samples['image']
        img_embeds, atts_img = self.encode_img(image)
        # img_embeds=paddle.to_tensor(np.load('img_embeds.npy'))
        # atts_img =paddle.to_tensor(np.load('atts_img.npy'))
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
        
        to_regress_tokens = self.llama_tokenizer(text, return_tensors='pd',
            padding='longest', truncation=True, max_length=self.max_txt_len,
            add_special_tokens=False)
        targets = masked_fill(to_regress_tokens.input_ids,to_regress_tokens
            .input_ids == self.llama_tokenizer.pad_token_id, -100)
        empty_targets = paddle.ones(shape=[atts_img.shape[0], atts_img.
            shape[1] + 1], dtype='int64').fill_(value=-100)
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
            to_regress_tokens.token_type_ids], axis=1)
        attention_mask=paddle.ones_like(attention_mask)
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
        import numpy as np
        ckpt={}
        ckpt_minigpt4 = paddle.load(ckpt_path)
        for name,values in ckpt_minigpt4.items():
            if 'weight' in name:
                ckpt[name]=values.T.astype("float32")
            # else:
        if ckpt_path:
            print('Load BLIP2-LLM Checkpoint: {}'.format(ckpt_path))
            msg = model.set_state_dict(state_dict=ckpt)
            
        llama_proj_bias={'llama_proj.bias':ckpt_minigpt4["llama_proj.bias"]}
        model.set_state_dict(state_dict=llama_proj_bias)
        weight_Qformer=paddle.load('blip2_pretrained.pdparams',return_numpy=True)
        for name,value in weight_Qformer.items():
            if name== 'query_tokens':
                model.query_tokens.set_value(weight_Qformer[name].astype("float32"))
            elif 'Qformer' in name and ('weight' in name or 'bias' in name):
                model.Qformer.state_dict()[name[8:]].set_value(weight_Qformer[name].astype("float32"))
        lm_head_weight=paddle.load("llama_model.lm_head.weight.pdparams")
        lm_head_weight['llama_model.lm_head.weight']=lm_head_weight['llama_model.lm_head.weight'].astype("float16")
        model.set_state_dict(state_dict=lm_head_weight)
        ln_vision_weight=paddle.load("ln_vision.pdparams")
        ln_vision_weight['ln_vision.weight']=ln_vision_weight['ln_vision.weight'].astype("float32")
        ln_vision_weight['ln_vision.bias']=ln_vision_weight['ln_vision.bias'].astype("float32")
        model.set_state_dict(state_dict=lm_head_weight)
        model.set_state_dict(state_dict=ln_vision_weight)
        return model
    
    #self.llama_proj.bias
def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)






def convert_weights_to_dtype(model, dtype: str):
    # trying to convert model dtype if necessary
    if dtype not in ["float16", "float32", "float64"]:
        raise ValueError("Not supported dtype: {}., only [float16, float32, float64] supported.".format(dtype))
    dtype_mapping = {
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
    }

    def convert_for_vit(layer):
        if isinstance(layer, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.nn.Conv2D)):
            if layer.weight.dtype != dtype_mapping[dtype]:
                layer.weight = transfer_param(layer.weight, restore_data=True, dtype=dtype)
                if paddle.any(paddle.isnan(layer.weight)):
                    print("vit", layer)
            if layer.bias is not None and layer.bias.dtype != dtype_mapping[dtype]:
                layer.bias = transfer_param(layer.bias, restore_data=True, dtype=dtype)
                if paddle.any(paddle.isnan(layer.weight)):
                    "vit", print(layer)

    def convert_for_common(layer):
        if hasattr(layer, "weight"):
            if layer.weight is not None and layer.weight.dtype != dtype_mapping[dtype]:
                
                layer.weight = transfer_param(layer.weight, restore_data=True, dtype=dtype)
                if paddle.any(paddle.isnan(layer.weight)):
                    print(layer)
        if hasattr(layer, "bias"):
            if layer.bias is not None and layer.bias.dtype != dtype_mapping[dtype]:
                layer.bias = transfer_param(layer.bias, restore_data=True, dtype=dtype)
                if paddle.any(paddle.isnan(layer.bias)):
                    paddle.print(layer)

    if isinstance(model, VisionTransformer):
        model.apply(convert_for_vit)
    else:
        model.apply(convert_for_common)
        
        
        


