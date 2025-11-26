"""
Prompt templates for RAG system.
"""

from typing import List, Dict


class PromptTemplateManager:
    """Manages prompt templates for different RAG tasks"""

    TEMPLATES = {
        'qa': """Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:""",

        'chat': """You are a helpful assistant.

Context:
{context}

User: {query}
