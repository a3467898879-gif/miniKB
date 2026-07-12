"""
LLM provider — streams chat completions from any OpenAI-compatible API.

Design: a single class that works with OpenAI, DeepSeek, Moonshot,
SiliconFlow, Ollama (with OpenAI shim), and any provider implementing
the /v1/chat/completions endpoint.

Streaming: yields text chunks as they arrive from the API, enabling
real-time SSE streaming to the frontend (same UX as ChatGPT).
"""

from typing import Generator

from openai import OpenAI

from config import settings


# System prompt template — instructs the LLM to answer from context only
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据以下检索到的参考资料回答用户的问题。

规则：
1. 只根据提供的参考资料回答，不要编造信息
2. 如果参考资料中没有相关内容，请明确告知用户
3. 回答要简洁、准确、有条理
4. 在回答末尾标注引用的资料来源编号，格式如 [1]、[2]
5. 如果问题与参考资料无关，可以正常对话但需说明"以下回答未使用知识库资料"

参考资料：
{context}"""


class LLMProvider:
    """Streams LLM responses from an OpenAI-compatible API."""

    def __init__(self):
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Lazily initialize the OpenAI client with configured credentials."""
        if self._client is None:
            if not settings.llm_api_key:
                raise ValueError(
                    "No LLM API key configured. Set LLM_API_KEY in .env"
                )
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
            )
        return self._client

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion tokens.

        Args:
            messages: OpenAI-format messages [{role, content}, ...]
            temperature: Lower = more deterministic, higher = more creative

        Yields:
            Text chunks as they arrive from the API.
        """
        client = self._get_client()
        stream = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        """Non-streaming chat completion (collects full response)."""
        return "".join(self.chat_stream(messages, temperature))

    def build_messages(
        self,
        question: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build the message list for the LLM API call.

        Structure:
          [system]  — RAG instructions + retrieved context
          [user]    — previous messages (conversation history)
          [assistant]
          ...
          [user]    — current question
        """
        # Format context from retrieved chunks
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                source = chunk.get("metadata", {}).get("document_name", "未知")
                context_parts.append(f"[{i}] 来源：{source}\n{chunk['content']}")
            context = "\n\n".join(context_parts)
        else:
            context = "（未检索到相关资料）"

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]

        # Add conversation history (last 6 messages for token efficiency)
        if history:
            messages.extend(history[-6:])

        messages.append({"role": "user", "content": question})
        return messages


# Singleton
llm_provider = LLMProvider()
