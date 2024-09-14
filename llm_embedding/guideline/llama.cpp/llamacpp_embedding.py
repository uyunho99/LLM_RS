import ast
from langchain.embeddings import LlamaCppEmbeddings

def get_embedding_for_text(model_path, input_text):
    embeddings = LlamaCppEmbeddings(model_path=model_path)

    if isinstance(input_text, list):
        # 문자열 리스트에 대한 임베딩 생성
        embedding_output = [embeddings.client.embed(text) for text in input_text]
    elif isinstance(input_text, str):
        # 단일 문자열에 대한 임베딩 생성
        embedding_output = embeddings.client.embed(input_text)
    else:
        raise ValueError("입력은 문자열 또는 문자열 리스트여야 합니다.")

    return embedding_output

if __name__ == "__main__":
    # 모델 경로와 텍스트 파일 경로를 변수로 지정
    model_path = "./Model/Llama-3-8B-Instruct-Gradient-1048k-Q4_K_M.gguf"
    file_path = "./Data/sample1.txt"

    # 파일에서 텍스트를 읽어옴
    with open(file_path, 'r', encoding='utf-8') as file:
        user_input = file.read()

    # 입력받은 텍스트가 리스트인지 문자열인지 판별
    try:
        # 입력을 리스트로 시도
        input_text = ast.literal_eval(user_input)
        if not isinstance(input_text, list) or not all(isinstance(item, str) for item in input_text):
            raise ValueError
    except (ValueError, SyntaxError):
        # 실패하면 문자열로 간주
        input_text = user_input

    # 입력받은 텍스트로부터 임베딩 생성
    embeddings = get_embedding_for_text(model_path, input_text)

    # 생성된 임베딩 출력
    if isinstance(input_text, list):
        print("각 요소별 임베딩 길이:", [len(e[0]) for e in embeddings])
        print("각 임베딩의 처음 5개 요소:", [e[0][:5] for e in embeddings])
    else:
        print("임베딩 길이:", len(embeddings[0]))
        print("임베딩의 처음 5개 요소:", embeddings[0][:5])