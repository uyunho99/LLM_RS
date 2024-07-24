from llama_index.embeddings.ollama import OllamaEmbedding
import ast

def get_embedding_for_text(input_text):
    ollama_embedding = OllamaEmbedding(
        model_name="llama3",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

    if isinstance(input_text, list):
        # 리스트일 경우 pass_embedding을 호출합니다.
        embedding = ollama_embedding.get_text_embedding_batch(input_text, show_progress=True)
    elif isinstance(input_text, str):
        # 문자열일 경우 query_embedding을 호출합니다.
        embedding = ollama_embedding.get_query_embedding(input_text)
    else:
        raise ValueError("입력은 문자열 또는 문자열 리스트여야 합니다.")

    return embedding

if __name__ == "__main__":
    # 텍스트 파일 경로를 변수로 지정합니다.
    file_path = "./Data/sample.txt"

    # 파일에서 텍스트를 읽어옵니다.
    with open(file_path, 'r', encoding='utf-8') as file:
        user_input = file.read()

    # 입력받은 텍스트가 리스트인지 문자열인지 판별합니다.
    try:
        # 입력을 리스트로 시도합니다.
        input_text = ast.literal_eval(user_input)
    except (ValueError, SyntaxError):
        # 실패하면 문자열로 간주합니다.
        input_text = user_input

    # 입력받은 텍스트로부터 Embedding을 생성합니다.
    embedding = get_embedding_for_text(input_text)

    # 생성된 Embedding을 출력합니다.
    if isinstance(input_text, list):
        print("Embedding 길이 (각 요소별):", [len(e) for e in embedding])
    else:
        print("Embedding 길이:", len(embedding))
