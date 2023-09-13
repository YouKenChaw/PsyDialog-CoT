import argparse

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

from dialogue.utils import print_args


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        writer = SummaryWriter(args.log_dir)
        print_args(args)

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, cache_dir=args.cache_dir)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                                 cache_dir=args.cache_dir)

    model.transformer.gradient_checkpoint = True
    assert model.transformer.gradient_checkpoint is True

    optimizer = torch.optim.AdamW(model.parameters, lr=args.learning_rate)

    train_set = PsyDialogueCotDataset(args.data_dir, tokenizer)
    train_loader = DataLoader(train_set, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True,
                              collate_fn=train_set.collate_fn)

    dev_set = PsyDialogueCotDataset(args.data_dir, tokenizer, data_type='dev')
    dev_loader = DataLoader(dev_set, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True,
                            collate_fn=train_set.collate_fn)

    num_training_steps = (len(train_loader) * args.n_epochs) // accelerator.gradient_accumulation_steps

    model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)

    global_step = 0

    model.train()
    for epoch in range(args.n_epochs):
        for batch_cnt, (input_ids, attention_mask, labels) in enumerate(train_loader):
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = output.loss

            metric(output.logits, labels, loss)

            accelerator.backward(loss)
            optimizer.step()

            if not accelerator.optimizer_step_was_skipped:
                pass

            global_step += 1

            if accelerator.is_main_process:
                accelerator.print()

            if global_step % args.log_steps == 0 and accelerator.is_main_process:
                writer.add_scalar('loss', train_loss, global_step=global_step)
                # writer.add_scalar('lr', )

            if global_step % args.eval_steps == 0:
                torch.cuda.empty_cache()
                model.eval()

                # val_metric =
                for input_ids, attention_mask, labels in dev_loader:
                    with torch.no_grad():
                        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

                    dev_metric(output.logits, labels, output.loss)



if __name__ == '__main__':
    args = setup_args()
    main(args)
