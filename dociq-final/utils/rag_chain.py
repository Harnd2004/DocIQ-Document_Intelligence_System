from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

SYSTEM_PROMPT = """You are a precise document analyst. Answer questions using ONLY the information in the provided context chunks.

Rules:
- Reason step-by-step before giving your final answer
- Always cite the source (e.g. "According to page 4...")
- If the answer is not in the context, respond: "This information is not present in the uploaded document."
- Be concise and direct

Few-shot examples:

Context: "The compound demonstrated 78% efficacy in Phase 2 trials across 200 participants."
Question: What was the efficacy rate?
Answer: The compound showed 78% efficacy in Phase 2 trials (200 participants). Source: Phase 2 results section.

Context: "All expense reports must be filed within 30 days of the transaction date."
Question: What is the expense report deadline?
Answer: Expense reports must be filed within 30 days of the transaction date. Source: Expense policy section.

Document Context:
{context}"""

HUMAN_PROMPT = """Question: {question}

Step-by-step reasoning:"""


def build_rag_chain(vectorstore, top_k: int = 5, model: str = "llama3.2"):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": top_k * 2},
    )

    llm = ChatOllama(model=model, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Chunk {i+1} | {d.metadata.get('source_file', 'unknown')} | p.{d.metadata.get('page', '?')}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def retrieve_chunks(retriever, question: str):
    return retriever.invoke(question)
