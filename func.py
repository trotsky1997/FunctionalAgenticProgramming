import orjson
import openai
from pydantic import BaseModel
from utils import get_schema_from_signature, get_return_schema_from_signature,gen_func_desc
from planning import Planer

class LLMbeforeFunc: # LLM before function,turn a natural input to a function input
    def __init__(self, func, openai_client):
        self.client = openai_client
        self.func = func
        # print(func)
        self.input_schema = get_schema_from_signature(func)
        self.planner = planner = Planer(openai_client)
        self.__name__ = self.func.__name__
        self.__doc__ = gen_func_desc(self.func)

    def user_decorate(self,prompt):
        return {"role": "user", "content": prompt}
    def assistant_decorate(self,ans):
        return {"role": "assistant", "content": ans}

    def __call__(self, natrual_input, rag_retrivel=[],tools=[],history=[],):
        if rag_retrivel != [] or tools != []:
            output,_ = self.planner(natrual_input,rag_retrivel=rag_retrivel,tools=tools,history=history)
        else:
            output = ''
        natrual_input = f"Info:{output}\nFunction name: {self.func.__name__}\nFunction docstrings: {self.func.__doc__}\n\n{natrual_input}"
        history += [self.user_decorate(natrual_input),]
        input_arguments = self.client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history,
        extra_body={
        "guided_json": self.input_schema,
        "guided_decoding_backend": "lm-format-enforcer"
        }).choices[0].message.content
        dict_body = orjson.loads(input_arguments)
        print(f'generate input arguments{dict_body} for {self.func.__name__}')
        return self.func(**dict_body)
    

class LLMafterFunc: # LLM after function,接受一个函数的返回值，返回一个自然语言的输出
    def __init__(self, func, openai_client):
        self.client = openai_client
        self.func = func
        self.input_schema = get_schema_from_signature(func)
        self.planner = planner = Planer(openai_client)
        self.__name__ = self.func.__name__
        self.__doc__ = gen_func_desc(self.func)

    def user_decorate(self,prompt):
        return {"role": "user", "content": prompt}
    def assistant_decorate(self,ans):
        return {"role": "assistant", "content": ans}

    def __call__(self, format_instruction, rag_retrivel=[],tools=[],history=[],**kwargs):
        function_output = self.func(**kwargs)
        
        format_instruction = f"Function name: {self.func.__name__}\nFunction docstrings: {self.func.__doc__}\nFunction return value:{function_output}:{type(function_output)}\n\n{format_instruction}"
        if rag_retrivel != [] or tools != []:
            output,_ = self.planner(format_instruction,rag_retrivel=rag_retrivel,tools=tools,history=history)
            format_instruction = f"Info:{output}\nFunction name: {self.func.__name__}\nFunction docstrings: {self.func.__doc__}\nFunction return value:{function_output}:{type(function_output)}\n\n{format_instruction}"
        else:
            format_instruction = f"Function name: {self.func.__name__}\nFunction docstrings: {self.func.__doc__}\nFunction return value:{function_output}:{type(function_output)}\n\n{format_instruction}"
        history += [self.user_decorate(format_instruction),]
        output = self.client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history,
        ).choices[0].message.content

        return output
    
class LLMpretendFunc:
    def __init__(self, func, openai_client,instruction=''):
        self.client = openai_client
        self.func = func
        self.planner = planner = Planer(openai_client)
        self.input_schema = get_schema_from_signature(func)
        self.output_schema = get_return_schema_from_signature(func)
        self.instructions = f'You need to pretend to be a function named {func.__name__},\n Function docstrings: {self.func.__doc__}\n\nAnd becareful to {instruction}'
        self.__name__ = self.func.__name__
        self.__doc__ = gen_func_desc(self.func)

    def user_decorate(self,prompt):
        return {"role": "user", "content": prompt}
    def assistant_decorate(self,ans):
        return {"role": "assistant", "content": ans}

    def __call__(self, rag_retrivel=[],tools=[],history=[],**kwargs):
        formated_kwargs = self.instructions + '\n' + " ".join([f'{k} is {v},' for k,v in kwargs.items()])
        if rag_retrivel != [] or tools != []:
            output,_ = self.planner(formated_kwargs,rag_retrivel=rag_retrivel,tools=tools,history=history)
            history += [self.user_decorate(f'Info:{output}\n'+formated_kwargs),]
        else:
            history += [self.user_decorate(formated_kwargs),]
        output = self.client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history,
        extra_body={
        "guided_json": self.output_schema,
        "guided_decoding_backend": "lm-format-enforcer"
        }).choices[0].message.content

        return output

def llm_wrapper(client,):
    def ret(func):
        return LLMbeforeFunc(func,openai_client=client)
    return ret

if __name__ == '__main__':
    from openai import OpenAI
    client = OpenAI(
        base_url='https://api.chemllm.org/v1',
        api_key="token-abc123",
    )

    def add(x:int,y:int) -> int:
        """
        The function "add" takes two integer inputs and returns their sum.
        
        :param x: The parameter `x` is an integer input for the `add` function
        :type x: int
        :param y: The parameter `y` in the `add` function is an integer type
        :type y: int
        :return: An integer value that is the sum of the two input parameters, x and y.
        """
        return x+y
    
    llm_add = LLMbeforeFunc(add,model_name="ChemLLM-20B-Chat-DPO",openai_client=client)

    a = llm_add('计算968+245',tools=[])
    print(a)

    llm_add_1 = LLMafterFunc(add,model_name="ChemLLM-20B-Chat-DPO",openai_client=client)
    b = llm_add_1(x=1,y=2,format_instruction="1+2等于几？",tools=[],history=[])

    print(b)

    class return_type(BaseModel):
        reason: str
        result: int

    def add(x:int,y:int) -> return_type:
        """
        The function "add" takes two integer inputs and returns their sum.
        
        :param x: The parameter `x` is an integer input for the `add` function
        :type x: int
        :param y: The parameter `y` in the `add` function is an integer type
        :type y: int
        :return: An integer value that is the sum of the two input parameters, x and y.
        """
        pass

    llm_pretend_add = LLMpretendFunc(add,model_name="ChemLLM-20B-Chat-DPO",openai_client=client,instruction='add two numbers')
    c = llm_pretend_add(x=1,y=2,tools=[],history=[])
    print(123)
    print(c)