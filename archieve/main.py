from typing import ForwardRef, List, Optional, Union
from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

class yes_enum(str, Enum):
    Yes = "Yes"
    No = "No"

class num_enum(str, Enum):
    one = 1
    two = 2
    three = 3

class house(BaseModel):
    price: int
    age: int
    address: str

class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int
    Yes: yes_enum
    num: num_enum
    pig: Optional[str]
    dog: Union[ForwardRef('house'),Optional[str]]
    home: ForwardRef('house')
    gain: List[ForwardRef('house')]
    # childs: List[ForwardRef('AnswerFormat')]
print(AnswerFormat.schema_json())

client = OpenAI(
    base_url='https://api.chemllm.org/v1',
    api_key="token-abc123",
    )

def chat(prompt,history=[]):
    print(history)
    prompt_message = {'role': 'user', 'content': prompt}
    response = client.chat.completions.create(
    model="AI4Chem/ChemLLM-20B-Chat-DPO",
    messages=history+[prompt_message],
    )
    return response.choices[0].message.content,history+[prompt_message,response.choices[0].message]


# def stylish_enum(prop, prompt):
#     pass

# def stylish_one(prop, prompt):
#     pass

# def chat(draft, instruct):
#     pass

# def gen_one(draft, schema, prop, prompt):
#     pass

# def gen_array(draft, schema, prop, prompt):
#     pass

# def gen_enum(draft, schema, prop, prompt):
#     pass

# def gen_obj(draft, schema, prop, prompt):
#     pass

# def gen(draft, schema, key, prompt):
#     if key not in schema.get('required'):
#         return ''
#     prop = schema.get('properties').get(key)
#     if '$ref' in prop:
#         ref_ = prop.get('$ref').split('/')[-1]

# # def gen_all(schema,prompt):
# #     ret = {}
# #     if schema.get('allOf') is not None:

# #     for key in schema.get('required'):
# #         res = gen(None, schema, key, prompt)
# #         ret[key] = res
# #     return ret

import json


def gen(schema, raw_prompt,history):
    DATA = {}

    def input_wrap(prompt,history):
        res,history = chat(prompt,history)
        return res,history

    def input_based_on_type(prompt, type, required=True,history=None):
        if type == "integer":
            while True:
                try:
                    res,history = input_wrap(prompt,history)
                    if res.isdigit():
                        return int(res)
                    else:
                        res,history = input_wrap("请输入一个整数",history)
                        return int(res)
                except:
                    continue
        elif type == "string":
            res,history = input_wrap(prompt,history)
            return res
        elif type == "boolean":
            while True:
                res,history = input_wrap(prompt + " (true/false): ",history)
                value = res.strip().lower()
                if value in ["true", "false"]:
                    return value == "true"
                res,history = input_wrap("请输入 'true' 或 'false'",history)
                if value in ["true", "false"]:
                    return value == "true"
        elif type is None and not required:
            value,history = input_wrap(prompt + " (可选，留空跳过): ")
            return value if value != "" else None

    def handle_enum(prompt, enum_options,history):
        options_str = ", ".join(enum_options)
        while True:
            value,history = input_wrap(f"{prompt} ({options_str}): ",history)
            if value in enum_options:
                return value
            value,history = input_wrap("输入的选项不在范围内，请重新输入。",history)
            if value in enum_options:
                return value

    def handle_anyOf(anyOf, prompt):
        prefix = f"选择 {prompt} 的一个类型:"
        for i, option in enumerate(anyOf, 1):
            prefix = '\n' +f"{i}. {option.get('type', 'Option ' + str(i))}"
        while True:
            choice,history = input_wrap(prefix+'\n'+"请选择一个类型编号: ",history)
            if choice.isdigit() and 1 <= int(choice) <= len(anyOf):
                local_schema = anyOf[int(choice) - 1]
                if local_schema['type'] == 'null':
                    return None
                local_schema['title'] = prompt
                local_schema = {'properties':{prompt:local_schema},'required':[prompt]}
                return parse_and_ask(local_schema)[prompt]
            choice,history = input_wrap("输入的编号不在范围内，请重新输入。",history)
            if choice.isdigit() and 1 <= int(choice) <= len(anyOf):
                local_schema = anyOf[int(choice) - 1]
                if local_schema['type'] == 'null':
                    return None
                local_schema['title'] = prompt
                local_schema = {'properties':{prompt:local_schema},'required':[prompt]}
                return parse_and_ask(local_schema)[prompt]

    def parse_and_ask(schema, data=None,defs=None,history=None):
        nonlocal DATA
        if data is None:
            data = {}
        if defs == None:
            defs = schema.get('$defs')

        required_fields = schema.get('required', [])
        
        if 'allOf' in schema:
            for item in schema.get('allOf'):
                subschema_name = item.get('$ref').split('/')[-1]
                subschema = schema.get('$defs').get(subschema_name)
                data.update(parse_and_ask(subschema, None,defs=defs))
        elif 'enum' in schema:
            return handle_enum(schema.get('title'), schema.get('enum'))


        
        for key, value in schema.get('properties', {}).items():
            required = key in required_fields
            if 'properties' in value:  # Handle nested objects
                # print(f"\n请输入 {key} 的信息：")
                if required :
                    switch = True
                else:
                    ans,_ = input_wrap(f"{raw_prompt}\n现在我们来生成一个JSON，我们已经生成的数据部分，{DATA}\n\n是否需要提供 {key} 数据? (yes/no): ",history)
                    if ans.lower() == 'yes':
                        switch = True
                    else:
                        switch = False
                if switch:
                    data[key] = parse_and_ask(value,history=history) 
                else:
                    data[key] = None
            elif '$ref' in value:  # Handle reference
                local_subschema_name = value['$ref'].split('/')[-1]
                local_subschema = defs.get(local_subschema_name)
                data[key] = parse_and_ask(local_subschema,None,defs=defs,history=history)

            elif 'anyOf' in value:  # Handle anyOf
                data[key] = handle_anyOf(value['anyOf'], key)
            elif 'items' in value:  # Handle arrays
                items = []
                continue_adding = input_wrap(f"{raw_prompt}\n现在我们来生成一个JSON，我们已经生成的数据部分，{DATA}\n\n是否添加 {key} 数据? (yes/no): ")[0].lower() == 'yes'
                while continue_adding:
                    item_data = parse_and_ask(defs.get(value['items']['$ref'].split('/')[-1]))
                    items.append(item_data)
                    continue_adding = input_wrap(f"{raw_prompt}\n现在我们来生成一个JSON，我们已经生成的数据部分，{DATA}{items}\n\n是否继续添加 {key} 数据? (yes/no): ")[0].lower() == 'yes'
                data[key] = items
            else:
                prompt = f"{raw_prompt}\n现在我们来生成一个JSON，我们已经生成的数据部分，{DATA}\n\n现在请输入 {key},它的类型是 {value.get('type')}"
                if required :
                    switch = True
                else:
                    ans,_ = input_wrap(f"{raw_prompt}\n现在我们来生成一个JSON，我们已经生成的数据部分，{DATA}\n\n是否需要提供 {key} 数据? (yes/no): ",history)
                    if ans.lower() == 'yes':
                        switch = True
                    else:
                        switch = False
                if switch:
                    data[key] = input_based_on_type(prompt, value.get('type'), required,history) 
                else:
                    data[key] = None 
        
            DATA.update(data)

        return data

    return parse_and_ask(schema,history=history)

def main():
    schema = json.load(open('./example.json'))
    print(schema)
    prompt = input('111')
    result = gen(schema, prompt,[])
    print("\n生成的 JSON 数据如下:")
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
