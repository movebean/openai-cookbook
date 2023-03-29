import os
import re
import requests
import pandas as pd
import tiktoken
import openai
import numpy as np
import threading
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from bs4 import BeautifulSoup
from collections import deque

data_dir = "data/"
domain = "note"

def extract_urls(text):
    # 匹配URL的正则表达式，可以匹配http和https，分隔符可以是空格、右括号或右方括号
    url_regex = r'(https?://\S+)[\s)\]]'

    # 使用正则表达式匹配文本中的所有URL
    urls = re.findall(url_regex, text)

    return [url[:-1] for url in urls]


def get_text_from_url(url):
    with open('text/'+ domain + "/" + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:
        try:
            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)
        except Exception as e:
            print("Exception: ", e)

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def fetch_data():
    # Create a queue to store the URLs to crawl
    links = deque([])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([])

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    if not os.path.exists("processed/" + domain):
            os.mkdir("processed/" + domain)

    # Get all the text files in the text directory
    for file in os.listdir(data_dir):
        with open(data_dir + file) as f:
            text = f.read()
            for link in extract_urls(text):
                if link not in seen:
                    links.append(link)
                    seen.add(link)

    lock = threading.Lock()
    threads = []
    for i in range(20):
        t = threading.Thread(target=_crawl, args=(lock, links))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

def _crawl(lock, links):
    lock.acquire()
    while links:
        link = links.pop()
        lock.release()
        try:
            get_text_from_url(link)
        except Exception as e:
            print("Exception: ", e)
        lock.acquire()
    lock.release()

df = None

def save_data():
    global df

    texts=[]

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):

        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[10:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/' + domain + '/scraped.csv')
    df.head()

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv('processed/' + domain + '/scraped.csv', index_col=0)
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))


# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text):
    max_tokens = 500
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

def do_work():
    global df

    fetch_data()

    save_data()

    max_tokens = 500
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

    ################################################################################
    ### Step 9
    ################################################################################

    tokenizer = tiktoken.get_encoding("cl100k_base")
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    ################################################################################
    ### Step 10
    ################################################################################

    # Note that you may run into rate limit issues depending on how many files you try to embed
    # Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv('processed/' + domain + '/embeddings.csv')
    df.head()

    df=pd.read_csv('processed/' + domain + '/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    df.head()
################################################################################
### Step 12
################################################################################

def create_context(
    question, df, max_len=3800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=3000,
    size="ada",
    debug=False,
    max_tokens=1000,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.ChatCompletion.create(
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
            messages=[
                {"role": "system", "content": "你是一个AI助手，用中文回答问题"},
                {"role": "user", "content": f"基于给定的上下文用中文回答问题, 如果基于给定上下文不能回答问题, 就说\"I don't know\"\n\n上下文: {context}\n\n---\n\n问题: {question}\n回答:"},
                ],
        )
        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result
    except Exception as e:
        print(e)
        return ""

do_work()

import socket

# 创建TCP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP地址和端口号
ip_address = '0.0.0.0'
port = 8888
server_socket.bind((ip_address, port))

# 监听连接请求
server_socket.listen()

print(f'Server is listening on {ip_address}:{port}...')

while True:
    # 接受客户端连接
    client_socket, address = server_socket.accept()
    print(f'Connection from {address} has been established!')

    while True:
        # 接收数据
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
            break

        # 处理数据
        messages = data.strip().split('\n')
        for message in messages:
            print(f'Received message: {message}')
            answer = answer_question(df, question=message)
            # 发送响应
            response = f'You sent: {message}\n Answer: {answer}\n'
            client_socket.send(response.encode('utf-8'))

    # 关闭客户端连接
    client_socket.close()

# 关闭服务器套接字
server_socket.close()

