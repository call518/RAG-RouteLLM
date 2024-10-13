import streamlit as st
from streamlit_chat import message
from streamlit_js_eval import streamlit_js_eval
import uuid
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import re
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

import os
import shutil

## (선택적) LangSmith 분석 연동
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["KEYS"]["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]= "RAG-Service"

## (필수) Pinecone 사용을 위해 필수
os.environ["PINECONE_API_KEY"] = st.secrets["KEYS"]["PINECONE_API_KEY"]

# 페이지 정보 정의
st.set_page_config(page_title="Play RAG", page_icon=":books:", layout="wide")
st.title(":books: _:red[Play RAG] OpenAI(Ollama) /w Pinecone_")

# Initialize session state with default values
default_values = {
    'session_id': str(uuid.uuid4()),
    'api_url': None,
    'source_document': None,
    'llm': None,
    'retriever': None,
    'chat_history_user': [],
    'chat_history_ai': [],
    'chat_history_rag_contexts': [],
    'store': {},
    'is_analyzing': False,
    'is_analyzed': False,
    'openai_api_key': None
}

def init_session_state():
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
        print(f"[session_state] {key}={st.session_state.get(key, '')}")
    os.system('rm -rf ./uploads/*')

init_session_state()
upload_dir = f"./uploads/{st.session_state['session_id']}"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

def reset_force():
    for key, value in default_values.items():
        st.session_state[key] = value
        print(f"[session_state] {key}={st.session_state.get(key, '')}")

# URL 패턴 정의 (기본적인 URL 형태를 검증)
url_pattern = re.compile(
    r'^(?:http|ftp)s?://'  # http:// 또는 https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # 도메인 이름
    r'localhost|'  # localhost
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4 주소
    r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # IPv6 주소
    r'(?::\d+)?'  # 포트 번호
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def is_valid_url(url):
    return re.match(url_pattern, url) is not None

# Pinecone 관련 설정
pc = Pinecone()
pinecone_index_name = 'rag-service'
def reset_pinecone(index_name=pinecone_index_name):
    pc.delete_index(index_name)

def read_txt(file):
    # upload_dir = f"./uploads/{st.session_state['session_id']}"
    loader = TextLoader(f"{upload_dir}/{file}")
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

def read_pdf(file):
    # upload_dir = f"./uploads/{st.session_state['session_id']}"
    loader = PyPDFLoader(f"{upload_dir}/{file}")
    docs = loader.load()
    # shutil.rmtree(upload_dir)
    return docs

# Contextualize question (질답 히스토리를 이용해, 주어진 질문을 문맥상 정확하고 독립적인 질문으로 보완/재구성 처리해서 반환)
contextualize_q_system_prompt = (
    "채팅 기록과 채팅 기록의 맥락을 참조할 수 있는 최신 사용자 질문이 주어지면 채팅 기록 없이도 이해할 수 있는 독립형 질문을 작성하세요. 질문에 대답하지 말고 필요한 경우 다시 작성하고 그렇지 않으면 있는 그대로 반환하십시오. (답변은 한국어로 작성하세요.)"
    # "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# QA 시스템 프롬프트 설정 (사용자 질문과, 벡터DB로부터 얻은 결과를 조합해 최종 답변을 작성하기 위한 행동 지침 및 질답 체인 생성)
system_prompt = (
    "당신은 질의 응답 작업의 보조자입니다. 다음 검색된 컨텍스트 조각을 사용하여 질문에 답하세요. 답을 모르면 모른다고 말하세요. 최소 3개의 문장을 사용하고 답변은 상세한 정보를 담되 간결하게 유지하세요. (답변은 한국어로 작성하세요.)"
    # "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#--------------------------------------------------

def main():
    # 사이드바 영역 구성
    with st.sidebar:
        st.title("Parameters")

        selected_ai = st.sidebar.radio("AI", ("OpenAI", "Ollama"))
        def update_api_url():
            if st.session_state['selected_ai'] == "OpenAI":
                st.session_state['api_url'] = 'https://api.openai.com/v1'
            elif st.session_state['selected_ai'] == "Ollama":
                st.session_state['api_url'] = 'http://localhost:11434'

        st.session_state['selected_ai'] = selected_ai
        update_api_url()

        if selected_ai == "OpenAI":
            st.session_state['openai_api_key'] = st.text_input("Enter OpenAI API Key:", type="password")
            if st.session_state['openai_api_key']:
                os.environ["OPENAI_API_KEY"] = st.session_state['openai_api_key']
            selected_llm = st.sidebar.selectbox("Select LLM:", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"])
        elif selected_ai == "Ollama":
            selected_llm = st.sidebar.selectbox("Select LLM:", ["gemma2", "llama3", "codegemma"])

        # 최종 답변 작성용 LLM 선택 옵션 제공
        if selected_ai == "OpenAI":
            if not st.session_state['openai_api_key']:
                st.warning("Please enter the OpenAI API Key.")
                st.stop()
            st.session_state['llm'] = ChatOpenAI(
                base_url=st.session_state['api_url'],
                model=selected_llm,
                temperature=0,
            )

            embeddings = OpenAIEmbeddings(
                base_url=st.session_state['api_url'],
                model="text-embedding-3-large" # dimension = 3072
            )
            dimension = 3072
        else:
            st.session_state['llm'] = Ollama(
                base_url=st.session_state['api_url'],
                model=selected_llm,
                temperature=0,
            )

            embeddings = OllamaEmbeddings(
                base_url=st.session_state['api_url'],
                model="mxbai-embed-large",  # dimension = 1024
            )
            dimension = 1024

        doc_type = st.radio("Document Type:", ("URL", "File Upload"))

        if doc_type == "URL":
            st.session_state['source_document'] = st.text_input("Please enter the document URL", value=st.session_state.get('url', ''))
        elif doc_type == "File Upload":
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                with open(f"{upload_dir}/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f"File uploaded: {uploaded_file.name}")
                st.session_state['source_document'] = uploaded_file.name

        # 사용자 선택 및 입력값을 기본으로 RAG 데이터 준비
        if st.button("Embedding"):
            st.session_state['is_analyzing'] = True

            docs = []
            
            if doc_type == "URL":
                # 주요 세션 정보 구성
                if not st.session_state['source_document']:
                    st.error("[ERROR] URL 정보가 없습니다.")
                    st.stop()

                if not is_valid_url(st.session_state['source_document']):
                    st.error("비정상 URL 입니다")
                    st.stop()

                # 문서 로드 및 분할
                loader = WebBaseLoader(
                    web_paths=(st.session_state['source_document'],),
                )
                docs = loader.load()
            
            if doc_type == "File Upload":
                if uploaded_file is not None:
                    if uploaded_file.type == "application/pdf":
                        docs = read_pdf(uploaded_file.name)
                    elif uploaded_file.type == "text/plain":
                        docs = read_txt(uploaded_file.name)
                    else:
                        st.error("Unsupported file type")
                        return
                
                # 파일 업로드 후, 업로드된 파일 삭제
                del uploaded_file
                shutil.rmtree(upload_dir)

            # 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            print(f"splits ====> 청크 개수: {len(splits)}")

            # 모든 청크 벡터화 및 메타데이터 준비 (for Pinecone에 임베딩)
            documents_and_metadata = []
            for i, doc in enumerate(splits):
                document = Document(
                    page_content=doc.page_content,
                    metadata={"id": i + 1, "source": st.session_state['source_document']}
                )
                documents_and_metadata.append(document)

            # Pinecone Index 초기화
            if pinecone_index_name in pc.list_indexes().names():
                pc.delete_index(pinecone_index_name)

            # Pinecone Index 생성
            pc.create_index(
                name=pinecone_index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

            # Pinecone Embedding 처리
            vectorstore = PineconeVectorStore.from_documents(
                documents_and_metadata,
                embeddings,
                index_name=pinecone_index_name
            )
            
            # 주어진 URL 문서 내용 처리(임베딩)
            st.session_state['retriever'] = vectorstore.as_retriever(search_type="similarity", k=5, score_threshold=0.8)
            if st.session_state['retriever']:
                st.success("Embedding 완료!")
            else:
                st.error("Embedding 실패!")
                st.stop()

            # RAG Chain 생성
            history_aware_retriever = create_history_aware_retriever(
                st.session_state['llm'],
                st.session_state['retriever'],
                contextualize_q_prompt
            )

            question_answer_chain = create_stuff_documents_chain(st.session_state['llm'], qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            st.session_state['store'] = {}
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state['store']:
                    st.session_state['store'][session_id] = ChatMessageHistory()
                return st.session_state['store'][session_id]

            st.session_state['conversational_rag_chain'] = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            st.session_state['is_analyzing'] = False
            st.session_state['is_analyzed'] = True

        if st.session_state.get('is_analyzed', False) == True:
            if st.button("Reset"):
                reset_force()
                streamlit_js_eval(js_expressions="parent.window.location.reload()")

#-----------------------------------------------------------------------------------------------------------

    # 메인 창 로딩 가능 여부(retriever 객체 존재) 확인
    try:
        if not (st.session_state['retriever'] and st.session_state['session_id']) or st.session_state['is_analyzing']:
            st.stop()
    except Exception as e:
        print(f"[Exception]: {e}")
        st.markdown("좌측 사이드바에서 필수 정보를 입력하세요.")
        st.stop()

    ## Container 선언 순서가 화면에 보여지는 순서 결정
    container_history = st.container()
    container_user_textbox = st.container()

    ### container_user_textbox 처리
    with container_user_textbox:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            result = st.session_state['conversational_rag_chain'].invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": st.session_state['session_id']}
                },
            )

            print(result)
            ai_response = result['answer']
            
            rag_contexts = result.get('context', [])
            print("RAG 데이터:")
            for context in rag_contexts:
                print(context)

            st.session_state['chat_history_user'].append(user_input)
            st.session_state['chat_history_ai'].append(ai_response)
            st.session_state['chat_history_rag_contexts'].append(rag_contexts)

    ### container_history 처리
    if st.session_state['chat_history_ai']:
        ### 디버깅...
        print("====================================")
        print(st.session_state["chat_history_user"])
        print(st.session_state["chat_history_ai"])
        print(st.session_state["chat_history_rag_contexts"])
        print("====================================")

        with container_history:
            for i in range(len(st.session_state['chat_history_ai'])):
                message(st.session_state["chat_history_user"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["chat_history_ai"][i], key=str(i))
                
                # 소스 데이터 표시
                rag_contexts = st.session_state["chat_history_rag_contexts"][i]
                if rag_contexts:
                    with st.expander("Show Contexts"):
                        for context in rag_contexts:
                            st.write(context)

#----------------------------------------------------

if __name__ == "__main__":
    # Error 메세지/코드 숨기기
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
