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
Assistant:""",
    }

    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Return a prompt template by name."""
        if template_name not in cls.TEMPLATES:
            available = ", ".join(sorted(cls.TEMPLATES))
            raise ValueError(f"Unknown prompt template '{template_name}'. Available: {available}")
        return cls.TEMPLATES[template_name]

    @classmethod
    def format_template(cls, template_name: str, **kwargs: str) -> str:
        """Format a named prompt template with keyword arguments."""
        return cls.get_template(template_name).format(**kwargs)
