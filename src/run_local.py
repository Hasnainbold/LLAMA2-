from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from src.helper import *
print("ok")
DEFAULT_SYSTEM_PROMPT="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of 
answering something not correct. If you don't know the answer to a question,
please don't share false information."""

B_INST ,E_INST = "[INST]" , "[/INST]"
B_SYS,E_SYS = "<<SYS>>\n","\n<</SYS>>\n\n"


instruction = "Convert the following text from English to Hindi: \n\n {text}"

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST


prompt = PromptTemplate(template=template, input_variables=["text"])


llm = CTransformers(model='C:/Users/HASNAIN/Downloads/Generative ai/LLAMA2-/LLAMA2-/LLAMA2-/LLAMA2-/model/llama-2-7b-chat.ggmlv3.q4_0 (1).bin',
                    model_type='llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                   )


LLM_Chain=LLMChain(prompt=prompt, llm=llm)

print(LLM_Chain.run("how are you ?"))
