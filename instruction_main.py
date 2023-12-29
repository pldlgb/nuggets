import gc
import json
import logging
import os
import textwrap


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anchor import logger_root
from common import setup_env, mk_parser, AdvantageLogger, smart_tokenizer_and_embedding_resize
from models import build_model_signature, build_tokenizer, build_model
from models.meta_optimizer import AttnOptimWrapper
from tasks import load_task
from utils.logger import setup_logger, tabular_pretty_print
from utils.tools import ensure_folder

logger = logging.getLogger("task")


def the_shape(pack):
    if isinstance(pack, (list, tuple)):
        return f"{len(pack)} * {the_shape(pack[0])}"
    if isinstance(pack, torch.Tensor):
        return pack.size()

@torch.no_grad()
def do_infer_probs_zero(batched_choices_input):
    batched_choices_logprobs = []
    for batched_one_choice_input in batched_choices_input:
        batch_input_ids, batch_attention_mask, batch_choice_start, batch_choice_end = batched_one_choice_input
        bs = len(batch_input_ids)

        batched_logits = model(
            input_ids=batch_input_ids,  # [B, L']
            attention_mask=batch_attention_mask
        ).logits
        batched_output = F.log_softmax(batched_logits, dim=-1)  # [B, L', Vocab]

        batched_one_choice_logprobs = []
        for input_ids, choice_start, choice_end, lm_logprobs in zip(batch_input_ids, batch_choice_start, batch_choice_end, batched_output):
            choice_tokens = input_ids[choice_start:choice_end].unsqueeze(1)  # [L, 1]
            choice_logprobs = lm_logprobs[choice_start - 1 : choice_end - 1]  # [L, Vocab]

            extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)

            choice_length = choice_end - choice_start
            lm_log_p = torch.sum(extracted).item()
            norm_lm_log_p = (lm_log_p / choice_length).item()

            choice_lm_info = {"lm_log_p": lm_log_p, "norm_lm_log_p": norm_lm_log_p}
            batched_one_choice_logprobs.append(choice_lm_info)
        batched_choices_logprobs.append(batched_one_choice_logprobs)
    return batched_choices_logprobs

@torch.no_grad()
def do_infer_probs(exemplar_attn_kv, exemplar_attn_mask, batched_choices_input):
    batched_choices_logprobs = []
    for batched_one_choice_input in batched_choices_input:
        batch_input_ids, batch_attention_mask, batch_choice_start, batch_choice_end = batched_one_choice_input
        bs = len(batch_input_ids)

        merged_attn_mask = torch.cat((exemplar_attn_mask.expand(bs, -1), batch_attention_mask), dim=1)
        if args.model_type == "bloom":
            # [B*#Heads, Length, Hidden]
            def _expand(t, target_size):
                _bs, _head, _len, _hidden = 1, *t.size()
                return t.reshape(_bs, _head, _len, _hidden).expand(target_size * _bs, -1, -1, -1).reshape(target_size * _bs * _head, _len, _hidden)

            expand_exemplar_attn_kv = [[_expand(layer_k, bs), _expand(layer_v, bs)] for layer_k, layer_v in exemplar_attn_kv]
        else:
            # [B, #Heads, Length, Hidden]
            expand_exemplar_attn_kv = [[layer_k.expand((bs, -1, -1, -1)), layer_v.expand((bs, -1, -1, -1))] for layer_k, layer_v in exemplar_attn_kv]

        batched_logits = model(
            input_ids=batch_input_ids,  # [B, L']
            attention_mask=merged_attn_mask,  # [B, L + L']
            past_key_values=expand_exemplar_attn_kv,  # num_layers * 2 * [B, num_heads, L, H]
        ).logits
        batched_output = F.log_softmax(batched_logits, dim=-1)  # [B, L', Vocab]

        batched_one_choice_logprobs = []
        for input_ids, choice_start, choice_end, lm_logprobs in zip(batch_input_ids, batch_choice_start, batch_choice_end, batched_output):
            choice_tokens = input_ids[choice_start:choice_end].unsqueeze(1)  # [L, 1]
            choice_logprobs = lm_logprobs[choice_start - 1 : choice_end - 1]  # [L, Vocab]

            extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)

            choice_length = choice_end - choice_start
            lm_log_p = torch.sum(extracted).item()
            norm_lm_log_p = (lm_log_p / choice_length).item()

            choice_lm_info = {"lm_log_p": lm_log_p, "norm_lm_log_p": norm_lm_log_p}
            batched_one_choice_logprobs.append(choice_lm_info)
        batched_choices_logprobs.append(batched_one_choice_logprobs)
    return batched_choices_logprobs


if __name__ == "__main__":
    parser = mk_parser()
    args = parser.parse_args()

    if args.debug:
        logger_root = logger_root.joinpath("DEBUG")

    logger_root = logger_root.joinpath("main")
    dataset_name = args.dataset
    logger_folder = logger_root.joinpath(dataset_name)

    task_name = f"seed{args.seed}_main{args.kv_iter}"
    task_name += f"_{args.prompt_version}"
    task_name += f"_{args.model_type}_{args.model_size}"
    task_name += f"_{args.exemplar_method}{'' if args.exemplar_method == 'written' else args.num_k_shots}"
    task_name += f"_eps{args.step_size}_beta{args.momentum}"

    setup_env(gpu_s=args.gpus, seed=args.seed)
    ensure_folder(logger_folder, parents=True)
    setup_logger(
        logger_folder,
        log_file_name=f"{task_name}.log",
        console_output=not args.no_console,
    )

    logger.info(f"Task Prepared: {task_name}")
    logger.info(f"\tDataset: {dataset_name}")
    logger.info(f"\tLogger save at {logger_folder}")

    # 1. load model, tokenizer
    model_signature = build_model_signature(args.model_type, args.model_size, args.model_path)
    tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side="right", model_path=args.model_path)
    model = build_model(args.model_type, args.model_size, args.in_8bit, model_path=args.model_path)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "[PAD]"},
        tokenizer=tokenizer,
        model=model,
    )
    torch.autograd.set_grad_enabled(False)
    logger.info(f"Model loaded: {model_signature}")

    # 2. load dataset (with demonstrations)
    TaskHandler = load_task(dataset_name)
    task_agent = TaskHandler(args.prompt_version, args.prompt_path, args.test_path)
    task_agent.set_seed(args.seed)
    task_agent.do_load()

    dataset = task_agent.mk_result_dataset(tokenizer, args)

    logger.info(f"Selected batch_size: {args.batch_size}")

    loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=2)

    logger.info("Running ...")

    # Zero Shot Forward
    generated_zero_info = []
    for batch_input in tqdm(loader, desc=f"Zero Shot Forward"):
        batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
        # batch_input = [[e for e in batch_choice] for batch_choice in batch_input]
        batch_output = do_infer_probs_zero(
            batch_input,
        )  # [batch_of_choice0, batch_of_choice1, ...]
        zipped_zero_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
        generated_zero_info.extend(zipped_zero_logprobs)
        
    # Set demonstrations
    if args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    elif args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_k_shots)
    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_k_shots)
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    # Demonstrations Slice
    # logger.info("before slice : ", len(exemplar_str))
    exemplar_str = exemplar_str[:int(len(exemplar_str)*args.num_prompt)]
    # logger.info("after slice : ", len(exemplar_str))

    text_width = os.get_terminal_size().columns - 30
    
    rate_dict = {}
    score_dict = {}
    start = args.start
    end = min(start+args.pace, len(exemplar_str))
    logger.info(str(start)+"----------------"+str(end))

    for i in tqdm(range(start, end)):
        rate_dict[i] = []
        exemplar_input_ids, exemplar_attn_mask = [e.cuda() for e in dataset.tokenize_demonstration(exemplar_str[i])]
        meta_optim = AttnOptimWrapper(model, args.model_type, step_size=args.step_size, momentum=args.momentum)
        meta_optim.init()

        trace_logger = AdvantageLogger()

        for idx in range(args.kv_iter):
            exemplar_kv = meta_optim.step(exemplar_input_ids)

            generated_info = []  # question * [choice0_prob, choice1_prob]
            for batch_input in tqdm(loader, desc=f"idx={idx}"):
                batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
                # batch_input = [[e for e in batch_choice] for batch_choice in batch_input]
                batch_output = do_infer_probs(
                    exemplar_kv,
                    exemplar_attn_mask.unsqueeze(0),
                    batch_input,
                )  # [batch_of_choice0, batch_of_choice1, ...]
                zipped_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
                generated_info.extend(zipped_logprobs)

            rate, metric, score = task_agent.post_process(generated_info, metric_output=False, generated_zero_info=generated_zero_info)
            rate_dict[i].append(rate[0])
            score_dict[i] = [list(i) for i in score]
            metric_s = json.dumps(metric, indent=None)
            logger.info(f"Iter={idx+1: <3} | {metric_s}")
            # trace_logger.submit(idx + 1, metric["lm_log_p"])
            # gc.collect()

        # for line in trace_logger.pretty_print():
        #     logger.info(line)
    
    json_data = json.dumps(rate_dict)

    # 将JSON字符串写入文件
    with open(f'{args.save_path}/{start}_{end}_score.json', 'w') as file:
        file.write(json_data)
    
    score_data = json.dumps(score_dict)
    with open(f'{args.save_path}/{start}_{end}_raw_score.json', 'w') as file:
        file.write(score_data)

