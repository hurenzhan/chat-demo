import os
import re, wordninja
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 递归字符分割器
from langchain.retrievers import TFIDFRetriever  # 最传统的关键字加权检索
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

_ = load_dotenv(find_dotenv())  # 加载 .env 到环境变量

# 配置 OpenAI 服务
openai.api_key = os.getenv('OPENAI_API_KEY')  # 设置 OpenAI 的 key
openai.api_base = os.getenv('OPENAI_API_BASE')  # 指定代理地址


# 预处理字符全都连在一起的行
def preprocess(text):
    def split(line):
        tokens = re.findall(r'\w+|[.,!?;%$-+=@#*/]', line)
        return [
            ' '.join(wordninja.split(token)) if token.isalnum() else token
            for token in tokens
        ]

    lines = text.split('\n')
    for i, line in enumerate(lines):
        if len(max(line.split(' '), key=len)) >= 20:
            lines[i] = ' '.join(split(line))
    return ' '.join(lines)


def read_pdf(path, start_page, end_page):
    """
    根据页数范围读取 PDF 文件并返回分页后的文本列表
    :param path: PDF 文件路径
    :param start_page: 开始页数
    :param end_page: 结束页数
    :return: 分割后的文本列表
    """

    # 加载文件
    loader = PyPDFLoader(path)
    # 分割文件
    pages = loader.load_and_split()

    # return [preprocess(page.page_content) for page in pages[start_page - 1:end_page]]
    return [preprocess(page.page_content) for page in pages[start_page - 1:end_page]]


def get_relevant_documents(stringList, user_query):
    """
    根据用户的查询，返回相关的文档
    :param stringList: 读取的文本列表
    :param user_query: 用户的查询
    :return: 相关的文档
    """

    # 从文档中创建分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # 每个 chunk 的长度
        chunk_overlap=60,  # 因为有些词会被分割开，比如“中国”
        length_function=len,  # 用于计算长度的函数
        add_start_index=True,  # 是否在每个 chunk 前面加上索引
    )

    # 取一个有信息量的章节
    paragraphs = text_splitter.create_documents(stringList)

    # 从文档中创建检索器
    # retriever = TFIDFRetriever.from_documents(paragraphs)

    # Facebook 的开源向量检索引擎
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(paragraphs, embeddings)

    # return retriever.get_relevant_documents(user_query)
    return db.similarity_search(user_query)


# 拼接文档列表
def concat_docs_list(list):
    result = ''

    for item in list:
        result += item.page_content

    return result


# 获取问答模板
def get_chat_template():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "你是看书机器人，你根据书本内容回答用户问题。\n" +
                "书本内容：\n{information}\n\n不要编造消息，字数要多，内容要详细，多举例子"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )


# 根据给定资料，生成问答
def get_chat_response(information, query):
    # 问答模板
    template = get_chat_template()

    # 问答模型
    llm = ChatOpenAI(temperature=0)

    # 生成问答
    response = llm(
        template.format_messages(
            information=information,
            query=query
        )
    )

    return response.content


# 读取 PDF 文件
pdf_text = read_pdf("wg史 北大马会编.pdf", 1, 377)

# 用户查询关键字
user_query = "文化大革命发生了什么，真相是什么？积极影响有哪些？"

# 检索文档
docs = get_relevant_documents(pdf_text, user_query)

# 拼接文档
doc_test = concat_docs_list(docs)

# print(doc_test, 'doc_test')

# 问答
content = get_chat_response(doc_test, user_query)
print(content)
