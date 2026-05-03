from langchain_ollama import OllamaLLM

def get_llm(llm_model="llama3"):

    llm = OllamaLLM(
        model=llm_model
    )

    return llm