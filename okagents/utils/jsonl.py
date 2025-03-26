import json


def add_to_jsonl(file_path, data):
    """data 应该要是 python dict 格式!!!!!"""
    # 如果是 json，转换为 python dict
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            raise ValueError("Input string is not a valid JSON object")
    # 将python dict转换为 json 存储
    with open(file_path, 'a+', encoding="utf-8") as file:
        json_line = json.dumps(data)
        file.write(json_line + '\n')
