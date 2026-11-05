
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from ingest import (
    BASE_DIR,
    DOCUMENTS_DIR,
    get_collection,
    ingest_documents,
    retrieve_context,
)

# Load environment variables from project folder
load_dotenv(
    dotenv_path=BASE_DIR / ".env",
    override=True
)

# Streamlit configuration
st.set_page_config(
    page_title="Indala College | Admission Assistant",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM DESIGN: NAVY BACKGROUND, LIGHT TEXT, RIGHT SIDEBAR
# --------------------------------------------------

st.markdown("""
<style>
/* Main page background */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #101B35 0%, #172B4D 100%);
    color: #F1F5F9;
}

/* Place Streamlit's sidebar on the RIGHT */
[data-testid="stAppViewContainer"] {
    display: flex;
    flex-direction: row-reverse;
}

[data-testid="stMain"] {
    direction: ltr;
    min-width: 0;
}

/* Main content text */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #F4C95D !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] {
    color: #F1F5F9;
}

/* Right sidebar */
section[data-testid="stSidebar"] {
    background-color: #223A5E;
    border-left: 1px solid #3B5578;
    border-right: none;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F4C95D !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #F1F5F9 !important;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea,
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #D4A72C !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton button {
    background-color: #D4A72C !important;
    color: #101B35 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton button:hover {
    background-color: #F4C95D !important;
    color: #101B35 !important;
}

/* Chat message panels */
[data-testid="stChatMessage"] {
    background-color: #223A5E;
    border: 1px solid #3B5578;
    border-radius: 12px;
    padding: 12px;
}

/* Keep chat text readable on the dark message panels. */
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #F1F5F9 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #223A5E;
    border: 1px solid #3B5578;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    background-color: #223A5E !important;
    color: #F1F5F9 !important;
}

[data-testid="stExpander"] summary *,
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
    color: #F1F5F9 !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #223A5E !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] p {
    color: #F1F5F9 !important;
}

/* Links and captions */
[data-testid="stMain"] a {
    color: #F4C95D !important;
}

[data-testid="stMain"] small,
[data-testid="stSidebar"] small {
    color: #D7E0EC !important;
}
</style>
""", unsafe_allow_html=True)

# System instructions for the admission chatbot
SYSTEM_PROMPT = """
You are the admission assistance chatbot for Indala College.

Answer using only the supplied college document excerpts.

If the information is unavailable in the excerpts, clearly say:
"Sorry, this information is not available in the supplied college documents.
Please contact the Indala College Admission Office for accurate information."

Do not invent fees, eligibility rules, admission dates, course details
or contact information.

Do not use outside knowledge or assumptions.
Use simple, polite, student-friendly language.
If the excerpts do not support an answer, say the information is unavailable.
Do not create or guess page numbers or document names.
"""

# Configuration
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.75"))
TOP_K = int(os.getenv("TOP_K", "5"))

UNAVAILABLE = (
    "Sorry, this information is not available in the supplied college "
    "documents. Please contact the Indala College Admission Office "
    "for accurate information."
)


# --------------------------------------------------
# GROQ API ANSWER GENERATION
# --------------------------------------------------

def generate_answer(question, matches):

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Please add your Groq API key to the .env file."
        )

    # Prepare retrieved document context
    context_parts = []

    for item in matches:
        context_parts.append(
            f"Source: {item['source']} | Page: {item['page']}\n"
            f"Excerpt: {item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Connect to Groq
    client = Groq(api_key=api_key)

    # Generate response
    response = client.chat.completions.create(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-20b"
        ),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Supplied college document excerpts:\n"
                    f"{context}\n\n"
                    "Answer only if the excerpts contain enough evidence. "
                    "Do not use external knowledge."
                )
            }
        ]
    )

    return response.choices[0].message.content.strip()


# --------------------------------------------------
# SAVE UPLOADED PDF DOCUMENTS
# --------------------------------------------------

def save_uploaded_files(uploaded_files):

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    saved = []

    for uploaded_file in uploaded_files:

        if not uploaded_file.name.lower().endswith(".pdf"):
            continue

        destination = (
            DOCUMENTS_DIR / Path(uploaded_file.name).name
        )

        destination.write_bytes(
            uploaded_file.getbuffer()
        )

        saved.append(destination)

    return saved


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🎓 Indala College")

st.subheader("AI Admission Assistant")

st.write(
    "Ask questions about courses, eligibility, fees, "
    "admission procedures and required documents. "
    "Answers are grounded in the college PDFs."
)


# --------------------------------------------------
# SIDEBAR - DOCUMENT ADMINISTRATION
# --------------------------------------------------

with st.sidebar:

    st.header("Document administration")

    st.caption(
        "Upload official college PDF documents, "
        "then process them."
    )

    uploaded_files = st.file_uploader(
        "Upload college PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Save PDF files
    if st.button(
        "Save uploaded PDFs",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please select at least one PDF.")

        else:

            saved = save_uploaded_files(uploaded_files)

            st.success(
                f"Saved {len(saved)} PDF file(s) "
                "in the documents folder."
            )

    # Process PDF documents
    if st.button(
        "Process documents",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Extracting, chunking and embedding PDF text..."
            ):

                result = ingest_documents()

            st.success(
                f"Processing complete: "
                f"{result['stored_chunks']} chunks stored. "
                f"ChromaDB now contains "
                f"{result['collection_count']} chunks."
            )

            if result["skipped_files"]:

                st.warning(
                    "No selectable text found in: "
                    + ", ".join(result["skipped_files"])
                    + ". These may be scanned PDFs and need OCR."
                )

        except Exception as exc:

            st.error(f"Processing failed: {exc}")

    st.divider()

    # Database status
    st.subheader("Database status")

    try:

        st.metric(
            "Stored text chunks",
            get_collection().count()
        )

    except Exception:

        st.metric("Stored text chunks", 0)

    st.caption(
        "Only upload official documents that the college "
        "is authorized to share."
    )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View source documents"):

                for source in message["sources"]:

                    st.write(
                        f"**{source['source']}** "
                        f"— Page {source['page']}"
                    )


# --------------------------------------------------
# USER QUESTION AND ANSWER
# --------------------------------------------------

question = st.chat_input(
    "Ask an admission-related question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            # Retrieve relevant PDF document chunks
            with st.spinner(
                "Searching official college documents..."
            ):

                matches = retrieve_context(
                    question,
                    top_k=TOP_K
                )

            # Filter irrelevant document chunks
            relevant_matches = [
                item
                for item in matches
                if item["distance"] <= MAX_DISTANCE
            ]

            if not relevant_matches:

                answer = UNAVAILABLE
                sources = []

            else:

                # Generate answer using Groq
                with st.spinner(
                    "Generating answer using Groq AI..."
                ):

                    answer = generate_answer(
                        question,
                        relevant_matches
                    )

                # Collect unique source documents and pages
                unique_sources = []
                seen = set()

                for item in relevant_matches:

                    key = (
                        item["source"],
                        item["page"]
                    )

                    if key not in seen:

                        seen.add(key)

                        unique_sources.append(
                            {
                                "source": item["source"],
                                "page": item["page"]
                            }
                        )

                sources = unique_sources

            # Display answer
            st.markdown(answer)

            # Display source references
            if sources:

                with st.expander("View source documents"):

                    for source in sources:

                        st.write(
                            f"**{source['source']}** "
                            f"— Page {source['page']}"
                        )

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

        except Exception as exc:

            error_message = (
                f"Unable to generate an answer: {exc}"
            )

            st.error(error_message)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": []
                }
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "For official confirmation of admission details, "
    "please contact the Indala College Admission Office."
)
