import copy
import os
import json

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_dir, tokenizer, data_type='train'):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.data = []
        self.no_loss_spans = []
        self.load_data()

    def load_data(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        batch_input_ids, batch_attn_mask, batch_label = [], [], []
        for input_ids, attn_mask, label in batch:
            batch_input_ids.append(input_ids)
            batch_attn_mask.append(attn_mask)
            batch_label.append(label)
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True,
                                                          padding_value=self.tokenizer.eos_token_id)
        batch_attn_mask = torch.nn.utils.rnn.pad_sequence(batch_attn_mask, batch_first=True, padding_value=0).to(
            torch.bool)
        batch_label = torch.nn.utils.rnn.pad_sequence(batch_label, batch_first=True, padding_value=-100)
        return batch_input_ids, batch_attn_mask, batch_label


class PsyDialogueCotDataset(BaseDataset):
    def __init__(self, data_dir, tokenizer, data_type='train'):
        super().__init__(data_dir, tokenizer, data_type)

    def load_data(self):
        data_file = os.path.join(self.data_dir, f'psy_{self.data_type}_data')
        no_loss_spans_file = os.path.join(self.data_dir, f'psy_{self.data_type}_no_loss_spans')
        if os.path.exists(data_file) and os.path.exists(no_loss_spans_file):
            self.data = torch.load(data_file, map_location='cpu')
            self.no_loss_spans = torch.load(no_loss_spans_file, map_location='cpu')
        else:
            with open(os.path.join(self.data_dir, f'psy_{self.data_type}.json'), 'r') as f:
                samples = json.load(f)
                f.close()
            for sample in samples:
                meta_instruction = '我想让你担任一位心理咨询师，使用理性情绪行为疗法和ABC理论，标注当前的患者情况以及需要采取的下一步策略，请你为患者进行问诊，'
                instruction_ids = self.tokenizer.encode(meta_instruction)
                assert isinstance(instruction_ids, list) and len(instruction_ids) > 0
                input_ids = copy.deepcopy(instruction_ids)
                no_loss_spans = [(0, len(instruction_ids))]
                for line in sample:
                    cur_no_loss_spans = []
                    cur_turn_ids = self.tokenizer.encode(line)
                    assert isinstance(cur_turn_ids, list) and len(cur_turn_ids) > 0
                    if len(input_ids + cur_turn_ids) > 2048:
                        break
                    input_ids.extend(cur_turn_ids)
                    no_loss_spans.extend(cur_no_loss_spans)
                if len(input_ids) == len(instruction_ids):
                    continue
                assert 0 < len(input_ids) <= 2048
                self.data.append(input_ids)
                self.no_loss_spans.append(no_loss_spans)
            # torch.save(self.data, data_file)
            # torch.save(self.no_loss_spans, no_loss_spans_file)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data[index])
        no_loss_spans = copy.deepcopy(self.no_loss_spans[index])
        data = torch.tensor(data, dtype=torch.long)
        attn_mask = torch.ones_like(data, dtype=torch.bool)
        label = copy.deepcopy(data)
        for no_loss_span in no_loss_spans:
            label[no_loss_span[0]: no_loss_span[1]] = -100
        return data, attn_mask, label
