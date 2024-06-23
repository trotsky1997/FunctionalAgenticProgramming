from typing import Literal, Union, List
from pydantic import BaseModel, Field

class knowledgebank_operator(BaseModel):
    '''
    Operator to query knowledgebanks.
    '''
    intention: str = Field(..., description="intnention of the operator, which, why to query the knowledgebank.")
    target_knowledgebank: Literal{knowledgebanks} = Field(..., description="Which Knowledge you want to query")
    hint_for_query: str = Field(..., description="hints for generate the query.")

class QA_operator(BaseModel):
    '''
    Operator for managing question-answering operations.
    '''
    intention: str = Field(..., description="Purpose of the QA operation.")
    attach_history: bool = Field(..., description="Whether to include past interactions in the current operation.")
    zero_temperature: bool = Field(..., description="Whether to generate deterministic responses with no randomness.")
    hint_for_query: str = Field(..., description="Hints to help formulate the QA query.")

class tool_function_operator(BaseModel):
    '''
    Operator for invoking specific tool functions.
    '''
    intention: str = Field(..., description="Intention behind using the tool function.")
    target_function: Literal{functions} = Field(..., description="The specific function to be invoked.")
    hint_for_arguments_generation: str = Field(..., description="Hints to assist in generating arguments for the function.")

class python_function_code_generation_operator(BaseModel):
    '''
    Operator to generate Python function code based on specified requirements.
    '''
    intention: str = Field(..., description="Purpose of generating Python function code.")
    hint_for_query: str = Field(..., description="Hints to guide the generation of Python code.")

class output_structirize_operator(BaseModel):
    '''
    Operator to structure the output in a specified format.
    '''
    intention: str = Field(..., description="Purpose of structuring the output.")
    format_requirement: str = Field(..., description="Required format for the output.")

class bool_branch_control_operator(BaseModel):
    '''
    Operator for controlling branching logic based on boolean conditions.
    If the condition is true, jump to the next operator; if the condition is false, jump to the next operator.z
    '''
    intention: str = Field(..., description="Purpose of controlling the branch logic.")
    branch_condition_hint: str = Field(..., description="Hints to define the condition for branching.")

class actionList_element(BaseModel):
    '''
    Element of an action list in a structured plan, capable of performing various operations.
    '''
    intention: str = Field(..., description="Intention behind the action element.")
    action: Union[knowledgebank_operator, QA_operator, tool_function_operator, python_function_code_generation_operator, output_structirize_operator, bool_branch_control_operator] = Field(..., description="The specific action to be performed as part of the plan.")

class plan(BaseModel):
    '''
    Structured plan consisting of a series of actionList elements.
    '''
    intention: str = Field(..., description="Overall purpose of the plan.")
    actionList: List[actionList_element] = Field(..., description="List of actions to be performed as part of the plan.")
