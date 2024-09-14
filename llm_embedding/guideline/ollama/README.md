# Ollama Embedding

## Ollama 설치 방법

### macOS

1. [Ollama 다운로드 페이지](https://ollama.com/download)에서 macOS용 설치 파일을 다운로드합니다.
2. 다운로드한 파일을 실행하여 설치 과정을 완료합니다.

### Windows (프리뷰)

1. [Ollama 다운로드 페이지](https://ollama.com/download)에서 Windows용 프리뷰 설치 파일을 다운로드합니다.
2. 다운로드한 파일을 실행하여 설치 과정을 완료합니다.

### Linux

1. 터미널을 열고 다음 명령어를 입력하여 Ollama를 설치합니다:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
2. 설치가 완료되면 Ollama가 정상적으로 설치되었는지 확인합니다:
    ```bash
    ollama --version
    ```

## Ollama Embedding 실행 방법 (by llamaindex)

1. Conda environment를 활성화합니다:
    ```bash
    conda activate [Conda environment name]
    ```

2. 필요한 라이브러리를 설치합니다:
    ```bash
    pip install -r requirements.txt
    ```

3. 터미널을 열고 다음 명령어를 입력하여 Ollama를 실행합니다:
    ```bash
    ollama run llama3
    ```

4. Ollama Embedding을 실행하기 전, 데이터 위치를 지정합니다:
    ```python
    file_path = "./Data/sample.txt"
    ```

5. 다음 명령어를 입력하여 Ollama Embedding을 실행합니다:
    ```bash
    python ollama_embedding.py
    ```