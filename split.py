import jsonlines

def extract_ids_to_file(input_file, output_file):
    with jsonlines.open(input_file) as reader, open(output_file, 'w') as writer:
        for obj in reader:
            writer.write(str(obj['id']) + '\n')

extract_ids_to_file('data/csqa/train_rand_split.jsonl', 'data/inhouse_split_qids.txt')