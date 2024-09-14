# LLM for embedding

## Introduction

LLM을 이용한 embedding 방법론을 소개합니다.

1) Huggingface

Huggingface에서 모델을 불러와서 text를 embedding하는 방법입니다.
GPU 메모리가 필요하여, 로컬에서는 구현이 불가능합니다.

    ```python
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_text_embedding(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        return embeddings.squeeze().numpy()
    ```

2) llama.cpp

llama.cpp는 C++로 LLM을 구현하여, 로컬 CPU에서 사용할 수 있습니다.
다만 양자화된 모델을 사용하므로, 정확도가 떨어질 수 있습니다.
llamacpp_embedding.py의 경우에는 langchain 라이브러리를 사용합니다.

    ```python
    embeddings = LlamaCppEmbeddings(model_path=model_path)
    embedding_output = embeddings.client.embed(input_text)
    ```

3) ollama

ollama은 Mac과 Linux 기반으로 로컬에서 LLM을 사용할 수 있는 프레임워크입니다.
마찬가지로 양자화된 모델을 사용하므로, 정확도가 떨어질 수 있습니다.
ollama_embedding.py의 경우에는 llama_index 라이브러리를 사용합니다.

    ```python
    ollama_embedding = OllamaEmbedding(
        model_name="llama3",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

    embedding = ollama_embedding.get_query_embedding(input_text)
    ```

