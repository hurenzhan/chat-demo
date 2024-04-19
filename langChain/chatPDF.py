from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import re, wordninja


# 预处理字符全都连在一起的行
# def preprocess(text):
#     def split(line):
#         tokens = re.findall(r'\w+|[.,!?;%$-+=@#*/]', line)
#         return [
#             ' '.join(wordninja.split(token)) if token.isalnum() else token
#             for token in tokens
#         ]
#
#     lines = text.split('\n')
#     for i, line in enumerate(lines):
#         if len(max(line.split(' '), key=len)) >= 20:
#             lines[i] = ' '.join(split(line))
#     return ' '.join(lines)


loader = PyPDFLoader("冯友兰《中国哲学史》.pdf")  # 加载文件
pages = loader.load_and_split()  # 分割文件

print(pages[0].page_content)  # 打印第一页的内容

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=200,  # 每个 chunk 的长度
#     chunk_overlap=50,  # 因为有些词会被分割开，比如“中国”
#     length_function=len,  # 用于计算长度的函数
#     add_start_index=True,  # 是否在每个 chunk 前面加上索引
# )
#
# paragraphs = text_splitter.create_documents([pages[0].page_content])
# for para in paragraphs:
#     print(para.page_content)
#     print('-------')