from copy import deepcopy
from typing import Literal, Union
import orjson
from pydantic import BaseModel
from func import LLMbeforeFunc


meta_code = """
from typing import Literal, Union, List
from pydantic import BaseModel, Field, conlist

class step(BaseModel):
    intention: str
    action: Literal{actions}

"""

def create_plan_schema(knowledgebanks=['IUPAC手册','数学rag知识库','化学rag知识库'],functions=['谷歌搜索','简易计算器','维基百科']):
    code = meta_code.format(actions=knowledgebanks+functions)
    namespace = {}
    exec(code, namespace)
    return namespace['step'].model_json_schema()


class Planer:
    def __init__(self, openai_client):
        self.client = openai_client
        self.max_iter = 3

    def user_decorate(self,prompt):
        return {"role": "user", "content": prompt}
    def assistant_decorate(self,ans):
        return {"role": "assistant", "content": ans}

    def __call__(self, natrual_input, rag_retrivel=[],tools=[],history=[],):
        actions = rag_retrivel + tools
        for rag in rag_retrivel:
            natrual_input = f"RAG name: {rag.__name__}\nRAG docstrings: {rag.__doc__}\n{natrual_input}"
        for tool in tools:
            natrual_input = f"Function name: {tool.__name__}\nFunction docstrings: {tool.__doc__}\n{natrual_input}"
        natrual_input = f"special name: `End`\n{natrual_input}"
        self.input_schema = create_plan_schema(knowledgebanks=[rag.__name__ for rag in rag_retrivel],functions=[tool.__name__ for tool in tools]+['End'])
        self.actions_map = {rag.__name__:rag for rag in rag_retrivel+tools}
        self.actions_map['End'] = None
        history += [self.user_decorate(natrual_input),]
        for i in range(self.max_iter):
            action = self.client.chat.completions.create(
            model="AI4Chem/ChemLLM-20B-Chat-DPO",
            messages=history,
            extra_body={
            "guided_json": self.input_schema,
            "guided_decoding_backend": "lm-format-enforcer"
            }).choices[0].message.content
            history += [self.assistant_decorate(action),]
            action_dict = orjson.loads(action)
            print(action_dict)
            if action_dict.get('action') == 'End':
                history += [self.user_decorate('End to stop.\nSummrize your observations.'),]
                break
            observe = self.actions_map.get(action_dict.get('action'))(''.join(action_dict.get('intention')),tools=tools,history=deepcopy(history))
            if i == self.max_iter - 1:
                history += [self.user_decorate(f'"observation":{observe}'+'\nSummrize your observations.'),]
                break
            else:
                history += [self.user_decorate(f'"observation":{observe}'),]
                print(123)

        output = self.client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history).choices[0].message.content

        return output

if __name__ == '__main__':
    from openai import OpenAI
    client = OpenAI(
        base_url='https://api.chemllm.org/v1',
        api_key="token-abc123",
    )

    def add0(x:int,y:int) -> int:
        """
        The `add` function takes two integer inputs `x` and `y`, and returns their sum as an integer.
        
        :param x: The parameter `x` is an integer input for the `add` function
        :type x: int
        :param y: The parameter `y` in the `add` function is an integer type
        :type y: int
        :return: The function `add` is returning the sum of the two input parameters `x` and `y`.
        """
        return x+y
    
    add = LLMbeforeFunc(add0, "AI4Chem/ChemLLM-20B-Chat-DPO", client)
    planer_0 = planer(client)
    print(planer_0(natrual_input="I want to add two numbers, one is 8,and another one is 3.",rag_retrivel=[add],tools=[add]))
