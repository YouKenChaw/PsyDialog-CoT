import jsonlines
import json


SPECIAL_TOKENS = {
    'Patient': '<|patient|>',
    'Doctor': '<|doctor|>',
    'System': '<|system|>',
    'Turn': '<|turn|>',
    'Stage': '<|stage|>',
    'Info': '<|info|>',
    'Summary': '<|summary|>',
    'Next': '<|next|>'
}


def preprocess(data_dir, save_dir, eos_token='</s>'):
    outputs = []
    with jsonlines.open(data_dir) as raw_data:
        for line in raw_data:
            dialog = line['dialog']
            cot = line['cot']
            if len(dialog) != len(cot):
                continue
            conversation = []
            for turn_id, utterance in enumerate(dialog):
                start_index = utterance.find('.')
                content = utterance[start_index + 1:].strip().split('\n')
                if len(content) != 2:
                    break
                patient_str, doctor_str = content[0].split('：'), content[1].split('：')
                assert (patient_str[0] == '患者' or len(patient_str) == 1) and doctor_str[0] == '治疗师', print(patient_str, doctor_str)
                patient_str = '<|patient|>' + patient_str[-1] + eos_token
                doctor_str = '<|doctor|>' + doctor_str[-1] + eos_token
                cot_str = cot[turn_id]
                assert int(cot_str['turn']) == turn_id + 1
                stage_str = '<|stage|>' + cot_str['stage'].strip() + eos_token
                info_str = '<|info|>' + cot_str['info'].strip() + eos_token
                summary_str = '<|summary|>' + cot_str['summary'].strip() + eos_token
                next_str = '<|next|>' + cot_str['next'].strip() + eos_token
                example = '<|turn|>' + str(cot_str['turn']) + eos_token + patient_str + stage_str + info_str + summary_str + next_str + doctor_str
                conversation.append(example)
            outputs.append(conversation)
    print(len(outputs))
    with open(save_dir, 'w', encoding='utf-8') as save_file:
        json.dump(outputs, save_file, ensure_ascii=False, indent=1)
        save_file.close()


def preprocess4finetune(data_dir, save_dir, eos_token='</s>'):
    outputs = []
    with jsonlines.open(data_dir) as raw_data:
        for line in raw_data:
            dialogue = line['dialog']
            history = []
            for utterance in dialogue:
                start_index = utterance.find('.')
                current_turn = utterance[start_index + 1:].strip().split('\n')
                if len(current_turn) != 2:
                    break
                patient_str, doctor_str = current_turn[0], current_turn[-1]
                if patient_str.split('：')[0] == '患者' and doctor_str.split('：')[0] == '治疗师':
                    patient_content = '<|patient|>' + patient_str.split('：')[-1] + eos_token
                    doctor_content = doctor_str.split('：')[-1] + eos_token
                    history.append(patient_content)
                    example = ''.join(history) + '<|doctor|>'
                    target = doctor_content
                    outputs.append({'inputs': example, 'label': target})
                    history.append('<|doctor|>' + doctor_content)
                else:
                    raise ValueError('Invalid data format.')
    print(len(outputs))
    with open(save_dir, 'w', encoding='utf-8') as save_file:
        json.dump(outputs, save_file, ensure_ascii=False, indent=1)
        save_file.close()


if __name__ == '__main__':
    preprocess4finetune('../../data/raw_data/psy_train.jsonl', '../../data/wo_cot/psy_train.json')
