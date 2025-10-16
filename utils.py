"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
############################################################
# ライブラリの読み込み
############################################################
import os
import sys
import logging
import unicodedata
import datetime
from dotenv import load_dotenv
import streamlit as st
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import LLMChain
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import SlackToolkit
from docx import Document
from sudachipy import tokenizer, dictionary

import constants as ct


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# PDF/TXT/DOCX 対応 RAG作成
############################################################
def create_customer_vectorstore():
    """
    顧客用PDF/TXT/DOCXからベクトルDBを作成
    """
    folder_path = "./data/rag/customer"
    docs_all = []

    if not os.path.exists(folder_path):
        print(f"⚠️ フォルダが存在しません: {folder_path}")
        return None

    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in ct.SUPPORTED_EXTENSIONS:
            loader_class = ct.SUPPORTED_EXTENSIONS[ext]
            loader = loader_class(file_path)
            docs = loader.load()
            docs_all.extend(docs)

    if not docs_all:
        print("⚠️ 顧客データが見つかりません。ベクトル化をスキップします。")
        return None

    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    split_docs = text_splitter.split_documents(docs_all)
    if not split_docs:
        print("⚠️ チャンク結果が空です。")
        return None

    embeddings = OpenAIEmbeddings()
    persist_dir = "./data/vectorstore/customer"
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✅ customer用ベクトルDB作成完了")
    return vectorstore


############################################################
# 既存 RAG Chain 作成関数
############################################################
def create_rag_chain(db_name):
    logger = logging.getLogger(ct.LOGGER_NAME)
    docs_all = []

    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    else:
        folder_path = ct.DB_NAMES[db_name]
        add_docs(folder_path, docs_all)

    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=f".db/{db_name}")

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_INQUIRY),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def add_docs(folder_path, docs_all):
    files = os.listdir(folder_path)
    for file in files:
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
            docs_all.extend(loader.load())


############################################################
# 顧客/会社/サービス用RAG実行関数
############################################################
def run_company_doc_chain(param):
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_service_doc_chain(param):
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_customer_doc_chain(param):
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_hr_doc_chain(query: str):
    ai_msg = st.session_state.hr_doc_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_faq_doc_chain(query: str):
    ai_msg = st.session_state.faq_doc_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]

def run_manual_doc_chain(query: str):
    ai_msg = st.session_state.manual_doc_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=ai_msg["answer"])])
    return ai_msg["answer"]


############################################################
# Slack通知用関数
############################################################
def notice_slack(chat_message):
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    for doc in docs + docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    docs_all = adjust_reference_data(docs, docs_history)

    docs_all_page_contents = [doc.page_content for doc in docs_all]

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs_all, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    bm25_retriever = BM25Retriever.from_texts(
        docs_all_page_contents,
        preprocess_func=preprocess_func,
        k=ct.TOP_K
    )
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=ct.RETRIEVER_WEIGHTS
    )

    employees = retriever.invoke(chat_message)
    context = get_context(employees)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)
    ])
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    messages = prompt_template.format_prompt(
        context=context, query=chat_message, format_instruction=format_instruction
    ).to_messages()
    employee_id_response = st.session_state.llm(messages)
    employee_ids = output_parser.parse(employee_id_response.content)

    target_employees = get_target_employees(employees, employee_ids)
    slack_ids = get_slack_ids(target_employees)
    slack_id_text = create_slack_id_text(slack_ids)
    context = get_context(target_employees)
    now_datetime = get_datetime()

    prompt = PromptTemplate(
        input_variables=["slack_id_text", "query", "context", "now_datetime"],
        template=ct.SYSTEM_PROMPT_NOTICE_SLACK,
    )
    prompt_message = prompt.format(slack_id_text=slack_id_text, query=chat_message, context=context, now_datetime=now_datetime)
    agent_executor.invoke({"input": prompt_message})

    return ct.CONTACT_THANKS_MESSAGE


############################################################
# 補助関数
############################################################
def adjust_reference_data(docs, docs_history):
    docs_all = []
    for row in docs:
        row_lines = row.page_content.split("\n")
        row_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_lines if ": " in item}
        employee_id = row_dict.get("従業員ID", "")
        same_employee_inquiries = []

        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_history_lines if ": " in item}
            if row_history_dict.get("従業員ID") == employee_id:
                same_employee_inquiries.append(row_history_dict)

        new_doc = Document()
        doc_text = ""
        if same_employee_inquiries:
            doc_text += "【従業員情報】\n" + "\n".join(row_lines) + "\n=================================\n"
            doc_text += "【この従業員の問い合わせ対応履歴】\n"
            for inquiry_dict in same_employee_inquiries:
                for key, value in inquiry_dict.items():
                    doc_text += f"{key}: {value}\n"
                doc_text += "---------------\n"
            new_doc.page_content = doc_text
        else:
            new_doc.page_content = row.page_content
        new_doc.metadata = {}
        docs_all.append(new_doc)
    return docs_all

def get_target_employees(employees, employee_ids):
    target_employees = []
    duplicate_check = []
    for employee in employees:
        num = employee.page_content.find("従業員ID")
        employee_id = employee.page_content[num+len("従業員ID")+2:].split("\n")[0]
        if employee_id in employee_ids and employee_id not in duplicate_check:
            duplicate_check.append(employee_id)
            target_employees.append(employee)
    return target_employees

def get_slack_ids(target_employees):
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find("SlackID")
        slack_id = employee.page_content[num+len("SlackID")+2:].split("\n")[0]
        slack_ids.append(slack_id)
    return slack_ids

def create_slack_id_text(slack_ids):
    return "と".join([f"「{id}」" for id in slack_ids])

def get_context(docs):
    context = ""
    for i, doc in enumerate(docs, start=1):
        context += "===========================================================\n"
        context += f"{i}人目の従業員情報\n"
        context += "===========================================================\n"
        context += doc.page_content + "\n\n"
    return context

def get_datetime():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y年%m月%d日 %H:%M:%S')

def preprocess_func(text):
    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text, mode)
    return list(set([token.surface() for token in tokens]))

def adjust_string(s):
    if type(s) is not str:
        return s
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
    return s

def build_error_message(message):
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])
