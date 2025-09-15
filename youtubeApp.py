
"""
Youtube captions + FAISS + LLM for Fantasy Football advice.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() #Loads API key from .env file

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document


CHUNK_SIZE = 1000 #Size of text chunk
CHUNK_OVERLAP = 150 #Overlap between chunks
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

#Extract the 11-character YouTube video ID from common URL formats.
def extract_video_id(url: str) -> str:
    pats = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
    ]
    for p in pats:
        m = re.search(p, url)
        if m:
            return m.group(1)

    if len(url) >= 11 and re.match(r"^[A-Za-z0-9_-]{11}$", url[-11:]):
        return url[-11:]
    raise ValueError("Couldn't parse a YouTube video ID from that URL.")

# Fetch English captions, if no captions RuntimeError is raised.
def get_captions_text(video_id: str) -> str:
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = transcripts.find_transcript(['en', 'en-US'])
        except Exception:
            t = transcripts.find_transcript(['en']).translate('en')
        entries = t.fetch()
    except (TranscriptsDisabled, NoTranscriptFound):
        raise RuntimeError("No captions available for this video.")
    text = "\n".join([e.get("text", "") for e in entries if e.get("text")])
    return text

#First time build or load a FAISS vector store for a video.
def build_or_load_vs(video_id: str):
    idx_dir = DATA_DIR / f"faiss_{video_id}"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if idx_dir.exists():
        vs = FAISS.load_local(str(idx_dir), embeddings, allow_dangerous_deserialization=True)
        return vs

    #Fetch captions, chunk, embed, and save index
    text = get_captions_text(video_id)
    #Split long text into overlapping chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    #Wrap chunks as LangChain Documents
    docs = [Document(page_content=c, metadata={"video_id": video_id}) for c in chunks]
    #Build FAISS from the chunk embeddings
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(idx_dir))
    return vs


FANTASY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant focused on FANTASY FOOTBALL.\n"
        "Only use information in the captions below. If it's not there, say you don't know.\n\n"
        "Captions:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely with actionable advice when relevant (start/sit, injury notes, PPR vs standard)."
    ),
)


def main():
    #API key checks
    if len(sys.argv) < 2:
        print("Usage: python simple_app.py <YouTube URL>")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY (e.g., in .env).")
        sys.exit(1)

    url = sys.argv[1]
    vid = extract_video_id(url)

    print(f"[load] building/loading FAISS for {vid} ...")
    vs = build_or_load_vs(vid)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vs.as_retriever(search_kwargs={"k": 4}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": FANTASY_PROMPT},
        return_source_documents=True,
    )

    print("\n Ask questions about this episode (type 'exit' to quit).\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not q:
            continue

        out = qa.invoke({"query": q})
        print("\nA:", out.get("result", "").strip(), "\n")

if __name__ == "__main__":
    main()
