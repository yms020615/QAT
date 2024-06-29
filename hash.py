import json
def replace_ids_and_save_to_text(file_path, output_jsonl_path, output_text_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(output_text_path, 'w') as text_file:
        for i in range(len(lines)):
            data = json.loads(lines[i])
            new_id = str(i + 1)
            data['id'] = new_id
            lines[i] = json.dumps(data)

            text_file.write(new_id + '\n')

    with open(output_jsonl_path, 'w') as jsonl_file:
        for line in lines:
            jsonl_file.write(line + '\n')

file_path = 'data/csqa/statement/train.statement.jsonl'
output_file_path = 'data/csqa/statement/train.statement2.jsonl'
output_text_path = 'data/csqa/inhouse_split_qids.txt'
replace_ids_and_save_to_text(file_path, output_file_path, output_text_path)