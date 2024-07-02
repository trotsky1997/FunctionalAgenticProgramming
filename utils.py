from functools import lru_cache
from typing import Any, Callable
import inspect
from typing import Callable, Any
import warnings

import astroid
import astypes
from openai import OpenAI
from pydantic import create_model


def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """

    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            if arg.default != inspect._empty:
                node = astroid.extract_node(str(arg.default))
                inferred_type = astypes.get_type(node)
                arguments[name] = (inferred_type, ...)
                continue
            else:
                arguments[name] = (Any, ...)
                continue
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)

    return model.model_json_schema()


def get_return_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    code = inspect.getsource(fn)

    signature = inspect.signature(fn)

    # Get the return annotation
    return_annotation = signature.return_annotation
    if return_annotation == inspect._empty:
        # If there's no return annotation, set it to Any
        return_annotation = Any
    return_dict = {f'return': (return_annotation, ...)} # of {fn.__name__}
    return create_model("return_dict", **return_dict).model_json_schema()


def user_decorate(prompt):
    return {"role": "user", "content": prompt}

def assistant_decorate(ans):
    return {"role": "assistant", "content": ans}

def history_seq_check(history):
    safe = True
    safe_roles = ['user', 'assistant']
    if len(history) == 0:
        return safe
    if history[0].get('role') == 'system':
        for i,item in enumerate(history[1:]):
            if item.get('role') != safe_roles[i%2]:
                safe = False
                break
    else:
        for i,item in enumerate(history):
            if item.get('role') != safe_roles[i%2]:
                safe = False
                break
    return safe

def safe_history_append(history, role, message):
    role_to_decorate = {'user': user_decorate, 'assistant': assistant_decorate}
    if history_seq_check(history):
        if len(history) == 0:
            if role == 'user':
                history += [role_to_decorate[role](message)]
            else:
                history = [{'role':'user','content':'hello!'}]
                history += [role_to_decorate[role](message)]
        else:
            if history[-1].get('role') != role:
                history += [role_to_decorate[role](message)]
            else:
                history[-1]['content'] += message
    else:
        raise ValueError('History sequence is not safe.')
    return history


client = OpenAI(
    base_url='https://api.chemllm.org/v1',
    api_key="token-abc123",
)

@lru_cache(1024)
def gen_func_desc(function):
    code = inspect.getsource(function)
    history = [{'role':'user','content':f'{code}\n\nGenerate a concise description for this python function to guide agent\'s tool choice.'}]
    desc = client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history
        ).choices[0].message.content
    print(desc)
    return desc