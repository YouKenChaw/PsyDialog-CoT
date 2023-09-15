import argparse
import math

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, set_seed

from dialogue.utils import print_args, SPECIAL_TOKENS, Metric
from dialogue.datasets import PsyDialogueCotDataset


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=29)
    parser.add_argument('--deepspeed', type=bool, default=False)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=2)
    parser.add_argument('--model_name_or_path', type=str, default='../.cache/baichuan2-7b-base')
    parser.add_argument('--cache_dir', type=str, default='../.cache')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default='1e-5')
    parser.add_argument('--warmup_rates', type=float, default=0.2)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--pad_vocab_size_to_multiple_of', type=int, default=16)
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print_args(args)

    if args.deepspeed:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({
        'pad_token': tokenizer.pad_token,
        'eos_token': tokenizer.eos_token,
    })
    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
    additional_special_tokens = list(set(additional_special_tokens + list(SPECIAL_TOKENS.values())))
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir)
    num_embs = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != num_embs or args.pad_vocab_size_to_multiple_of:
        p = args.pad_vocab_size_to_multiple_of
        target_size = len(tokenizer) if not p else math.ceil(len(tokenizer) / p) * p
        model.resize_token_embeddings(target_size)
    model.gradient_checkpoint = True
    assert model.gradient_checkpoint is True

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_set = PsyDialogueCotDataset(args.data_dir, tokenizer)
    train_loader = DataLoader(train_set, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True,
                              collate_fn=train_set.collate_fn)
    dev_set = PsyDialogueCotDataset(args.data_dir, tokenizer, data_type='dev')
    dev_loader = DataLoader(dev_set, batch_size=args.train_bsz_per_gpu, shuffle=False, drop_last=False,
                            collate_fn=train_set.collate_fn)

    num_training_steps = (len(train_loader) * args.n_epochs) // accelerator.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)

    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, dev_loader, scheduler)

    global_step = 0
    metric = Metric(device=torch.cuda.current_device())

    model.train()
    for epoch in range(args.n_epochs):
        for batch_cnt, (input_ids, attention_mask, labels) in enumerate(train_loader):
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()

            accelerator.backward(loss)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                scheduler.step()

            global_step += 1
            if global_step % args.log_steps == 0 and accelerator.is_main_process:
                accelerator.print(f'epoch: {epoch + 1}, current step: {batch_cnt}, total_step: {len(train_loader)}, skip: {accelerator.optimizer_step_was_skipped}, loss: {round(train_loss, 5)}, acc: {round(acc, 4)}, lr: {scheduler.get_last_lr()[0]}')

            if global_step % args.eval_steps == 0:
                torch.cuda.empty_cache()
                model.eval()

                dev_metric = Metric(torch.cuda.current_device())
                for input_ids, attention_mask, labels in dev_loader:
                    with torch.no_grad():
                        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

                    dev_metric(output.logits, labels, output.loss)
                dev_acc, dev_loss = dev_metric.get_metric()

                if accelerator.is_local_main_process:
                    accelerator.print(f'epoch: {epoch + 1}, step: {batch_cnt}, dev loss: {round(dev_loss, 5)}, dev acc: {round(dev_loss, 4)}')

                model.train()

            if global_step % args.save_steps == 0:
                model.save_checkpoint(args.output_dir, global_step)

    if global_step % args.save_step != 0:
        model.save_checkpoint(args.output_dir, global_step)


if __name__ == '__main__':
    args = setup_args()
    set_seed(args.random_seed)
    main(args)
