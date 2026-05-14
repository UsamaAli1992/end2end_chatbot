# RAG prompt: {context} + {question} → answer with grounding.
# Wording avoids over-refusal when only part of the context is relevant (common with small LLMs).
prompt_template = """
Use the following excerpts from the user's document library to answer the question.
- Prefer facts that appear in the excerpts. You may combine information from several excerpts.
- If the excerpts only partly address the question, answer what you can from them and keep the answer focused on that material.
- If the excerpts contain nothing relevant to the question, say you don't know based on the provided documents — do not invent citations or facts not supported by the excerpts.

Context:
{context}

Question: {question}

Only return the helpful answer below and nothing else (no preamble).
Helpful answer:
"""
