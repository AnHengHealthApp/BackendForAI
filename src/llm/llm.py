import os

from openai import OpenAI


class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, '..', '..', 'res', 'prompt.txt')

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def chat(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=messages,
        )

        content = response.choices[0].message.content
        return content.strip() if content else "模型未回傳任何內容。"
