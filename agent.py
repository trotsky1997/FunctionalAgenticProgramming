from func import LLMpretendFunc
from func import llm_wrapper
import tools


def agentfunc(query:str)->str:
    '''
    User input a function, and return a answer.
    '''
    pass


from openai import OpenAI
client = OpenAI(
    base_url='https://api.chemllm.org/v1',
    api_key="token-abc123",
)

tools_functions = [llm_wrapper(client)(tool) for tool in tools.all_functions]

agent = LLMpretendFunc(agentfunc,openai_client=client,instruction='use tools to answer questions.')

while True:
    q = input('Please input a question:')
    ans = agent(query=q,tools=tools_functions,history=[])
    print(ans)