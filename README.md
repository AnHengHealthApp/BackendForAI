# AnHengHealthApp AI Backend

This backend is based on the OpenAI API using Python.

## Settings

> `uv` is recommended for package management.

1. **Create a virtual environment.**

    * Using `uv`:

        ```shell
        uv venv
        ```

    * Using Python's built-in `venv`:

        ```shell
        python3 -m venv .venv
        ```

2. **Activate the virtual environment** (for non-`uv` users).

   ```shell
   source .venv/bin/activate
   ```

3. **Install dependencies.**

    * Using `uv`:

        ```shell
        uv sync
        ```

    * Using `pip`:

        ```shell
        pip install .
        ```

4. **Create the `.env` file.**

   Create a file named `.env` in the project root directory:

   ```dotenv
   # OpenAI API Settings
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   OPENAI_MODEL="gpt-4-turbo"
   
   # Embedding Model Settings
   MODEL_NAME="Qwen/Qwen3-Embedding-4B"
   EMBEDDING_PROVIDER="openai"
   ```

5. **Initialize data.**
    1. Place the `medicine_completion.jsonl` file into the `data/` folder.

       > You need to download the file from [here](https://drive.google.com/file/d/1c2L8HPKxhlp2-fj3K9_W_0jojdTYWdv3/view?usp=sharing) first.

    2. Run the Python script to initialize embeddings:

        * Using `uv`:

           ```shell
           uv run python utils/embedding.py
           ```

        * Using Python's built-in `venv`:

            ```shell
            python utils/embedding.py
            ```

## Run the Project

Run the following command to start the local server with auto-reload:

```shell
uvicorn src.api.app:app --reload
```
