from typing import Optional, List
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
import abc
import time
from PIL import Image
import os
import argparse
import ast
import os

from utils import ptp_utils, wo_utils
from utils.null_text_inversion import NullInversion
from losses.attn_loss import AttnLoss

# 设置扩散模型的调度器，定义了如何在扩散过程中调整噪声水平，包括起始和结束的beta值，以及beta值随步骤变化的调度方式。
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
LOW_RESOURCE = True
NUM_DDIM_STEPS = 50  # 指定在扩散过程中使用的步骤数量，决定了生成图像的细节程度和质量
GUIDANCE_SCALE = 7.5  # 控制引导扩散的强度，影响生成图像与文本提示的相关性
MAX_NUM_WORDS = 77  # 限制文本提示中词汇的最大数量


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

# 抽象基类，在图像生成过程中修改和控制模型的注意力矩阵
class AttentionControl(abc.ABC):

    # 在每一步生成之后调用的回调函数
    def step_callback(self, x_t):
        return x_t

    # 在每一步生成之间调用的函数
    def between_steps(self):
        return

    # 属性装饰器，返回无条件注意力层数，如果是低资源模式，则使用所有注意力层，否则为0
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    # 抽象方法，定义了如何处理注意力矩阵
    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    # __call__使得实例可以像函数一样被调用，根据当前注意力层的状态处理注意力矩阵
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # 如果当前的注意力层大于等于无条件注意力层的数量，则对注意力矩阵进行处理
        if self.cur_att_layer >= self.num_uncond_att_layers:
            # 调用forward方法处理注意力矩阵，根据是否是交叉注意力、所处的Unet位置进行特定处理
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        # 当当前的注意力层计数达到总的注意力层次数时
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    # 重置函数，用于重新开始一个生成过程前重置当前步骤和当前注意力层的状态
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    # 初始化函数，设置当前步骤和当前注意力层为0，注意力层数量为 - 1（待指定）
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):
    # "down_cross"、"mid_cross"、"up_cross"代表在U-Net架构的下采样、中间层、上采样阶段收集的交叉注意力矩阵，
    # "down_self"、"mid_self"、"up_self"则分别代表这三个阶段收集的自注意力矩阵。
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    # 实现了如何处理注意力矩阵的逻辑，这里简单地存储下来，避免内存开销过大
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)  # 收集不同类型的注意力矩阵
        return attn

    # 在每一步生成之间调用，用于更新注意力存储
    def between_steps(self):
        # 如果attention_store为空，直接使用当前步骤的step_store来初始化
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        # 遍历每个键值对，将step_store中收集的当前步骤的注意力矩阵累加到attention_store中相应的累积值上。这样做是为了跨不同生成步骤累积注意力矩阵的信息，以便进行后续分析或调整
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, token_indices: List[int], alpha: float, method: str, cross_retain_steps: float, n: int, iter_each_step: int, max_step_to_erase: int,
                 lambda_retain=1, lambda_erase=-.5, lambda_self_retain=1, lambda_self_erase=-.5):
        super(AttentionStore, self).__init__()
        # 初始化存储注意力矩阵的结构，以及其它控制参数
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.baseline = True
        # for suppression content
        self.ddim_inv = False
        self.token_indices = token_indices
        self.uncond = True
        self.alpha = alpha
        self.method = method  # default: 'soft-weight'
        self.i = None
        self.cross_retain_steps = cross_retain_steps * NUM_DDIM_STEPS
        self.n = n
        self.text_embeddings_erase = None
        self.iter_each_step = iter_each_step
        self.MAX_STEP_TO_ERASE = max_step_to_erase
        # lambdas of loss
        self.lambda_retain = lambda_retain
        self.lambda_erase = lambda_erase
        self.lambda_self_retain = lambda_self_retain
        self.lambda_self_erase = lambda_self_erase


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int=0):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_self_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select)
    images = []
    for i in range(attention_maps.shape[2]):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = image.transpose(2, 0, 1)
        images.append(image[None, :])
    images = np.concatenate(images)
    ptp_utils.save_image_grid(images, f'self-attn-map.png', grid_size=(16, 16))

def show_cross_attention(stable, prompt, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, save_name='cross-attn-map'):
    tokens = stable.tokenizer.encode(prompt)
    decoder = stable.tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), save_name=save_name)

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), save_name='self-attn-map-comp')

# Infernce Code
@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type='image'
):
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # 得到文本嵌入
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        scale = 20
    else:
        uncond_embeddings_ = None
        scale = 5

    # 初始化潜在向量
    latent, _ = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)

    _latent, _latent_erase = latent.clone().to(model.device), latent.clone().to(model.device)
    latents = torch.cat([_latent, _latent_erase])

    # 注意力损失函数 n是输入的文本嵌入（不含EOT）长度
    attn_loss_func = AttnLoss(model.device, 'cosine', controller.n, controller.token_indices,
                              controller.lambda_retain, controller.lambda_erase, controller.lambda_self_retain, controller.lambda_self_erase)

    model.scheduler.set_timesteps(num_inference_steps)
    # text embedding for erasing
    controller.text_embeddings_erase = text_embeddings.clone()

    # scale_range定义了一个范围，用于在生成过程中动态调整某些参数
    scale_range = np.linspace(1., .1, len(model.scheduler.timesteps))
    pbar = tqdm(model.scheduler.timesteps[-start_time:], desc='Suppress EOT', ncols=100, colour="red")
    for i, t in enumerate(pbar):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            if LOW_RESOURCE:
                context = (uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings)
        # 整合了无条件和条件文本嵌入，用于引导生成过程
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            if LOW_RESOURCE:
                context = (uncond_embeddings_, text_embeddings)
        controller.i = i

        # conditional branch: erase content for text embeddings
        if controller.i >= controller.cross_retain_steps:  # 这里执行软加权正则化
            controller.text_embeddings_erase = \
                wo_utils.woword_eot_context(text_embeddings.clone(), controller.token_indices, controller.alpha,
                                            controller.method, controller.n)

        controller.baseline = True
        if controller.MAX_STEP_TO_ERASE > controller.i >= controller.cross_retain_steps and not (controller.text_embeddings_erase == text_embeddings).all() and \
                (attn_loss_func.lambda_retain or attn_loss_func.lambda_erase or attn_loss_func.lambda_self_retain or attn_loss_func.lambda_self_erase):
            controller.uncond = False
            # 设置当前注意力层，跳过无条件注意力层
            controller.cur_att_layer = 32  # w=1, skip unconditional branch
            controller.attention_store = {}
            # 生成带有文本嵌入（未经过软加权正则化）的噪声预测样本
            noise_prediction_text = model.unet(_latent, t, encoder_hidden_states=text_embeddings)["sample"]
            # 计算当前的注意力图，用于分析哪些内容需要保留或擦除
            attention_maps = aggregate_attention(controller, 16, ["up", "down"], is_cross=True)
            self_attention_maps = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=False)

            del noise_prediction_text
            # update controller.text_embeddings_erase for some timestep
            iter = controller.iter_each_step
            while iter > 0:
                with torch.enable_grad():
                    controller.cur_att_layer = 32  # w=1, skip unconditional branch
                    controller.attention_store = {}
                    # conditional branch
                    text_embeddings_erase = controller.text_embeddings_erase.clone().detach().requires_grad_(True)
                    # forward pass of conditional branch with text_embeddings_erase
                    # 生成带有文本嵌入（经过软加权正则化）的噪声预测样本
                    noise_prediction_text = model.unet(_latent_erase, t, encoder_hidden_states=text_embeddings_erase)["sample"]
                    model.unet.zero_grad()
                    # 经过软加权正则化后的注意力图，用于与原始注意力图进行比较，优化擦除策略
                    attention_maps_erase = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=True)
                    self_attention_maps_erase = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=False)

                    # attention loss
                    loss = attn_loss_func(attention_maps, attention_maps_erase, self_attention_maps, self_attention_maps_erase)
                    if loss != .0:
                        pbar.set_postfix({'loss': loss if isinstance(loss, float) else loss.item()})
                        text_embeddings_erase = update_context(context=text_embeddings_erase, loss=loss,
                                                               scale=scale, factor=np.sqrt(scale_range[i]))
                    del noise_prediction_text
                    torch.cuda.empty_cache()
                    controller.text_embeddings_erase = text_embeddings_erase.clone().detach().requires_grad_(False)
                iter -= 1

        # "uncond_embeddings_ is None" for real images, "uncond_embeddings_ is not None" for generated images.
        context_erase = (uncond_embeddings[i].expand(*text_embeddings.shape), controller.text_embeddings_erase) \
            if uncond_embeddings_ is None else (uncond_embeddings_, controller.text_embeddings_erase)
        controller.attention_store = {}
        controller.baseline = False  # 关闭baseline模式
        # 结合无条件和条件文本嵌入，创建新的上下文
        # 这里创建了一个新的、更大的上下文张量。这个操作的目的是在单个扩散步骤中同时考虑这两组信息
        # 组合原始上下文和擦除上下文允许模型在执行扩散步骤时，同时考虑到保持某些特征不变（原始上下文）和修改或擦除特定特征（擦除上下文）的需求。
        contexts = [torch.cat([context[0], context_erase[0]]), torch.cat([context[1], context_erase[1]])]
        latents = ptp_utils.diffusion_step(model, controller, latents, contexts, t, guidance_scale, low_resource=LOW_RESOURCE)
        # 分别获取保留和擦除操作后的潜在向量
        _latent, _latent_erase = latents
        _latent, _latent_erase = _latent.unsqueeze(0), _latent_erase.unsqueeze(0)

    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def update_context(context: torch.Tensor, loss: torch.Tensor, scale: int, factor: float) -> torch.Tensor:
    """
    Update the text embeddings according to the attention loss.

    :param context: text embeddings to be updated
    :param loss: ours loss
    :param factor: factor for update text embeddings.
    :return:
    """
    grad_cond = torch.autograd.grad(outputs=loss.requires_grad_(True), inputs=[context], retain_graph=False)[0]
    context = context - (scale * factor) * grad_cond
    return context

def run_and_display(ldm_stable, prompts, controller, latent=None, generator=None, uncond_embeddings=None, verbose=True):
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                        num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                        generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t

def load_model(sd_version):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if sd_version == "sd_1_4":
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    elif sd_version == "sd_1_5":
        stable_diffusion_version = "runwayml/stable-diffusion-v1-5"
    else:
        raise ValueError('Unsupported stable diffusion version')

    ldm_stable = StableDiffusionPipeline.from_pretrained(stable_diffusion_version, scheduler=scheduler).to(device)
    # try:
    #     ldm_stable.disable_xformers_memory_efficient_attention()
    # except AttributeError:
    #     print("Attribute disable_xformers_memory_efficient_attention() is missing")
    return ldm_stable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='Real-Image', choices=['Generated-Image', 'Real-Image'], help='the type of remove')
    parser.add_argument('--inversion', type=str, default='NT', help='NT (Null-text), NPI (Negative-prompt-inversion)')
    parser.add_argument('--sd_version', type=str, default='sd_1_4', help='use sd_1_4 or sd_1_5')
    parser.add_argument('--seed', type=int, default=2, help='seed for generated image of stable diffusion')
    parser.add_argument('--prompt', type=str, default='A man with a beard wearing glasses and a hat in blue shirt', help='prompt for generated or real image')
    parser.add_argument('--image_path', type=str, default='./example_images/A man with a beard wearing glasses and a hat in blue shirt.jpg', help='image path')
    # 排除的词的索引
    parser.add_argument('--token_indices', type=ast.literal_eval, default='[[4,5],[7],[9,10]]', help='index for without word')  # List[int]
    # 定义在超过多少步骤后执行"wo"惩罚，用于控制生成过程的精细程度，默认值为0.2，这意味着在生成过程的前20%步骤中，模型将正常生成图像，而在之后的步骤中，将开始尝试抑制特定内容。
    parser.add_argument('--cross_retain_steps', type=ast.literal_eval, default='[.2,]', help='perform the "wo" punish when step >= cross_wo_steps')  # .0 == τ=1.0, .1 == τ=0.9, .2 == τ=0.8
    # 惩罚比率，用于调整内容的保留或抑制程度，默认为1.0，更高的值意味着更强的抑制效果
    parser.add_argument('--alpha', type=ast.literal_eval, default='[1.,]', help="punishment ratio")
    # 定义在扩散模型的最大步骤数进行内容抑制，默认为20
    parser.add_argument('--max_step_to_erase', type=int, default=20, help='erase/suppress max step of diffusion model')
    # 定义每个步骤更新文本嵌入的迭代次数，默认为10
    parser.add_argument('--iter_each_step', type=int, default=10, help="the number of iteration for each step to update text embedding")
    # 添加关于注意力保留和删除损失的λ值的参数，用于微调模型对特定内容的处理方式
    parser.add_argument('--lambda_retain', type=float, default=1., help='lambda for cross attention retain loss')
    parser.add_argument('--lambda_erase', type=float, default=-.5, help='lambda for cross attention erase loss')
    parser.add_argument('--lambda_self_retain', type=float, default=1., help='lambda for self attention retain loss')
    parser.add_argument('--lambda_self_erase', type=float, default=-.5, help='lambda for self attention erase loss')
    parser.add_argument('--method', type=str, default='soft-weight', help='soft-weight, alpha, beta, delete, weight')
    args = parser.parse_args()
    return args

## suppression for real image
def main(args, stable):
    """
    suppress content with EOT of text embeddings for realistic image.

    :param args:
    :param stable: stable diffusion model.
    :return: None.
    """

    del args.seed
    inversion = args.inversion
    image_path = args.image_path
    prompt = args.prompt
    print(args)

    # Null-text inversion
    null_inversion = NullInversion(stable)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, inversion=inversion, verbose=True)

    n = len(stable.tokenizer.encode(args.prompt))

    outdir = f'suppress_eot_results/{prompt}'
    os.makedirs(outdir, exist_ok=True)

    for token_indices in args.token_indices:
        for cross_retain_steps in args.cross_retain_steps:
            for alpha in args.alpha:
                print(f'|----------Suppress EOT (Real-Image): token_indices={token_indices}, alpha={alpha}, cross_retain_steps(1-tau)={cross_retain_steps}----------|')
                controller = AttentionStore(token_indices, alpha, args.method,  cross_retain_steps, n, args.iter_each_step, args.max_step_to_erase,
                                            lambda_retain=args.lambda_retain, lambda_erase=args.lambda_erase, lambda_self_retain=args.lambda_self_retain, lambda_self_erase=args.lambda_self_erase)
                tau = round(1.0 - cross_retain_steps, 3)
                image_inv, x_t = run_and_display(stable, [prompt], controller, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False)
                print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image, the suppressed image")
                ptp_utils.view_images([image_gt, image_enc, image_inv[0], image_inv[1]], save_name=f'{outdir}/{args.method}{alpha}-tau{tau}-token_idx{token_indices}-image')
                # show_cross_attention(stable, prompt, controller, 16, ["up", "down"], save_name=f'{outdir}/soft-weight{alpha}-tau{tau}-token_idx{token_indices}-attn')

## suppression for generated image
def main_gen(args, stable):
    """
    suppress content with EOT of text embeddings for generated images from stable diffusion model
    with given prompts.

    :param args:
    :param stable: stable diffusion model.
    :return: None.
    """

    del args.image_path
    prompt = args.prompt
    print(args)

    n = len(stable.tokenizer.encode(args.prompt))

    outdir = f'/data/zwd/suppress_eot_results_gen/{prompt}/seed{args.seed}'
    os.makedirs(outdir, exist_ok=True)

    # # 遍历抑制词索引、保留步骤比例和惩罚比率的组合
    for token_indices in args.token_indices:
        for cross_retain_steps in args.cross_retain_steps:
            for alpha in args.alpha:
                g_cpu = torch.Generator().manual_seed(args.seed)
                print(f'|----------Suppress EOT (Generated-Image): token_indices={token_indices}, alpha={alpha}, cross_retain_steps(1-tau)={cross_retain_steps}----------|')
                # 创建一个AttentionStore控制器，用于控制生成过程中的注意力机制
                controller = AttentionStore(token_indices, alpha, args.method, cross_retain_steps, n, args.iter_each_step, args.max_step_to_erase,
                                            lambda_retain=args.lambda_retain, lambda_erase=args.lambda_erase, lambda_self_retain=args.lambda_self_retain, lambda_self_erase=args.lambda_self_erase)
                # 计算tau值，用于后续计算
                tau = round(1.0 - cross_retain_steps, 3)
                image_inv, x_t = run_and_display(stable, [prompt], controller, latent=None, generator=g_cpu, uncond_embeddings=None, verbose=False)
                print("showing from left to right: the ground truth image, the suppressed image")
                ptp_utils.view_images([image_inv[0], image_inv[1]], save_name=f'{outdir}/{args.method}{alpha}-tau{tau}-token_idx{token_indices}-image')
                # show_cross_attention(stable, prompt, controller, 16, ["up", "down"], save_name=f'{outdir}/soft-weight{alpha}-tau{tau}-token_idx{token_indices}-attn')

if __name__=="__main__":
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    args = parse_args()
    stable = load_model(args.sd_version)
    if args.type == 'Real-Image':
        main(args, stable)
    elif args.type  == 'Generated-Image':
        main_gen(args, stable)
    else:
        raise ValueError('Unsupported type')
