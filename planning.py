from copy import deepcopy
from typing import Literal, Union
import orjson
from pydantic import BaseModel, Field
# from func import LLMbeforeFunc
from vectordb import Memory
from utils import user_decorate, assistant_decorate, safe_history_append


meta_tool_switch_prompt = """
You are a language model assistant. Your task is to help the user choose and call the most suitable function from the provided relevant documentation. Please read these documents carefully and select the most appropriate function based on the user's needs. Follow the steps below:

1. Read and understand all the provided function names.
2. Decide the need for tools based on the user's needs.

Relevant function names:
{context}
`End` for no need for tools.

User's need:
{query}

Based on the above information, please design a excution plan for information collection.

Return Format:
## Proposal

## Verification

## Plan

"""

meta_tool_selection_prompt = """
You are a language model assistant. Your task is to help the user choose and call the most suitable function from the provided relevant documentation. Please read these documents carefully and select the most appropriate function based on the user's needs. Follow the steps below:

1. Read and understand all the provided function documentation.
2. Choose the function that best fits the user's needs.

Relevant function documentation:
{context}

User's need:
{query}

Based on the above information, choose and call the most suitable function.
"""

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


class tool_switch(BaseModel):
    switch: Literal['Need for tools','No Need for tools']
    intention: str = Field(...,description='The intention for the tool using')
    

class ToolPlanning:
    def __init__(self, openai_client):
        self.client = openai_client
        self.swithch_jsonschema = tool_switch.model_json_schema()

    def __call__(self, natrual_input,rag_retrivel=[],tools=[],history=[],history_strategy='temp'):
        if history_strategy == 'temp':
            history = deepcopy(history)
        tool_list = []
        for rag in rag_retrivel:
            tool_list.append(f"RAG name: {rag.__name__}\nRAG docstrings: {rag.__doc__}")
        for tool in tools:
            tool_list.append(f"Function name: {tool.__name__}")

        tool_list.append("special name: `End` for stop or no need for tool use.")

        tool_list = "\n".join(tool_list)
        prompt = meta_tool_switch_prompt.format(context=tool_list, query=natrual_input)

        history = safe_history_append(history, 'user', prompt)
        plan = self.client.chat.completions.create(
                model="AI4Chem/ChemLLM-20B-Chat-DPO",
                messages=history,
                ).choices[0].message.content
                #         extra_body={
                # "guided_json": self.swithch_jsonschema,
                # "guided_decoding_backend": "lm-format-enforcer"
                # }
        history = safe_history_append(history, 'assistant', plan)
        # switch = orjson.loads(switch) 
        # switch_intention = switch.get('switch')
        # switch = switch.get('switch') == 'Need for tools'
        return plan.split('## Plan')[-1], history


class ToolSelect:
    def __init__(self, openai_client):
        self.client = openai_client
        self.memory = Memory(embedding_model='sentence-transformers/all-MiniLM-L6-v2')

    def __call__(self, natrual_input,rag_retrivel=[],tools=[],history=[],history_strategy='temp'):
        self.input_schema = create_plan_schema(knowledgebanks=[rag.__name__ for rag in rag_retrivel],functions=[tool.__name__ for tool in tools]+['End'])
        if history_strategy == 'temp':
            history = deepcopy(history)
        # tool_list = []
        # for rag in rag_retrivel:
        #     tool_list.append(f"RAG name: {rag.__name__}\nRAG docstrings: {rag.__doc__}")
        # for tool in tools:
        #     tool_list.append(f"Function name: {tool.__name__}")

        # tool_list.append("special name: `End` for stop or no need for tool use.")

        for rag in rag_retrivel:
            self.memory.save([f"RAG name: {rag.__name__}\nRAG docstrings: {rag.__doc__}"],[{'type':'rag'}],)
        for tool in tools:
            self.memory.save([f"Function name: {tool.__name__}\nFunction docstrings: {tool.__doc__}"],[{'type':'tool'}])

        self.memory.save(["special name: `End` for stop or no need for tool use."],[{'type':'special'}])

        related_docs = "\n\n".join([item['chunk'] for item in self.memory.search(query=natrual_input,top_n=3)]) + '\n\n' +"special name: `End` for stop or no need for tool use." 

        prompt = meta_tool_selection_prompt.format(context="\n\n".join(related_docs), query=natrual_input)
        history = safe_history_append(history, 'user', prompt)
        action = self.client.chat.completions.create(
                model="AI4Chem/ChemLLM-20B-Chat-DPO",
                messages=history,
                extra_body={
                "guided_json": self.input_schema,
                "guided_decoding_backend": "lm-format-enforcer"
                }).choices[0].message.content
        history = safe_history_append(history, 'assistant', action)
        action_dict = orjson.loads(action)
        return action_dict, history
    

class ReflexionStepExcutor:
    pass

class Planer:
    def __init__(self, openai_client):
        self.client = openai_client
        self.max_iter = 16
        self.swithch_jsonschema = tool_switch.model_json_schema()
        self.tool_planning = ToolPlanning(self.client)
        self.tool_select = ToolSelect(self.client)

    def user_decorate(self,prompt):
        return {"role": "user", "content": prompt}
    def assistant_decorate(self,ans):
        return {"role": "assistant", "content": ans}

    def __call__(self, natrual_input, rag_retrivel=[],tools=[],history=[],):
        self.actions_map = {rag.__name__:rag for rag in rag_retrivel+tools}
        self.actions_map['End'] = None
        # history += [self.user_decorate(natrual_input),]
        plan,_ = self.tool_planning(natrual_input,rag_retrivel=rag_retrivel,tools=tools,history=history)
        steps = [step for step in plan.split('\n') if not step.isspace() and step]
        print(steps)
        for i,step in enumerate(steps[:self.max_iter]):
            action_dict,history = self.tool_select(step,rag_retrivel=rag_retrivel,tools=tools,history=history)
            if action_dict.get('action') == 'End':
                history += [self.user_decorate('End to stop.\nSummrize your observations.'),]
                break
            print(f"To fullfill plan step {step}, calling tool {action_dict.get('action')} with intention {action_dict.get('intention')}")
            try:
                observe = self.actions_map.get(action_dict.get('action'))(''.join(action_dict.get('intention')),history=deepcopy(history))
            except Exception as e:
                observe = f"Error happened when calling tool {action_dict.get('action')}, error message: {e}"
            if i == self.max_iter - 1:
                history += [self.user_decorate(f'"Function name":{action_dict.get("action")},\n"observation":{observe}'+'\nSummrize your observations.'),]
                break
            else:
                history += [self.user_decorate(f'"Function name":{action_dict.get("action")},\n"observation":{observe}'),]

        output = self.client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history).choices[0].message.content

        return output,history

if __name__ == '__main__':
    pass
    # from openai import OpenAI
    # client = OpenAI(
    #     base_url='https://api.chemllm.org/v1',
    #     api_key="token-abc123",
    # )

    # def add0(x:int,y:int) -> int:
    #     """
    #     The `add` function takes two integer inputs `x` and `y`, and returns their sum as an integer.
        
    #     :param x: The parameter `x` is an integer input for the `add` function
    #     :type x: int
    #     :param y: The parameter `y` in the `add` function is an integer type
    #     :type y: int
    #     :return: The function `add` is returning the sum of the two input parameters `x` and `y`.
    #     """
    #     return x+y
    
    # add = LLMbeforeFunc(add0, "AI4Chem/ChemLLM-20B-Chat-DPO", client)
    # planer_0 = Planer(client)
    # print(planer_0(natrual_input="I want to add two numbers, one is 8,and another one is 3.",rag_retrivel=[add],tools=[add]))
