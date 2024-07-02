from openai import OpenAI
client = OpenAI(
    base_url='https://api.chemllm.org/v1',
    api_key="token-abc123",
)

def user_decorate(prompt):
    return {"role": "user", "content": prompt}
def assistant_decorate(ans):
    return {"role": "assistant", "content": ans}

history = [user_decorate('Hello, I am a user.')]

action = client.chat.completions.create(
        model="AI4Chem/ChemLLM-20B-Chat-DPO",
        messages=history,
        extra_body={
        "guided_json": self.input_schema,
        "guided_decoding_backend": "lm-format-enforcer"
        }).choices[0].message.content