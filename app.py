from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
### loaders
from langchain.document_loaders import PyPDFLoader, DirectoryLoader,TextLoader
### splits
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
### prompts
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from replicate.client import Client
from flask import Flask, request, render_template,jsonify
app = Flask(__name__)
chat_model = ChatOpenAI(model="gpt-4", temperature=0,openai_api_key="sk-proj-9iZEnF2h1ex63Vz8xj7sT3BlbkFJ46femQhpvksr53AQBjAf")
chat_model_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0,openai_api_key="sk-proj-9iZEnF2h1ex63Vz8xj7sT3BlbkFJ46femQhpvksr53AQBjAf")
class RAG:
    temperature = 0,
    top_p = 0.95,
    repetition_penalty = 1.15

    # splitting
    split_chunk_size = 800
    split_overlap = 0

    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

    # similar passages
    k = 5

    # paths
    txts_path = ''
    Persist_directory = 'uae-vectordb'

### download embeddings model
embeddings = embeddings = OpenAIEmbeddings(openai_api_key="sk-proj-9iZEnF2h1ex63Vz8xj7sT3BlbkFJ46femQhpvksr53AQBjAf")

### load vector DB embeddings
vectordb = FAISS.load_local(
   RAG.Persist_directory + '/faiss_index_hp',
    embeddings,
    allow_dangerous_deserialization=True
)

prompt_template = """
Please use only the given piece of context below and do not answer from your own knowledge.
Gave the answer in proper complete sentence.
{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template,
    input_variables = ["context", "question"]
)

retriever = vectordb.as_retriever(search_kwargs = {"k": RAG.k, "search_type" : "similarity"})

qa_chain = RetrievalQA.from_chain_type(
    llm = chat_model,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever,
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)
qa_chain_3_5 = RetrievalQA.from_chain_type(
    llm = chat_model_3_5,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever,
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)
def llm_ans(query):
    try:
      llm_response = qa_chain.invoke(query)
      ans = llm_response['result']
    except Exception as e:
        # Handle any errors and return the error message
        error_message = f"An error occurred: {str(e)}"
        return error_message,

    return ans
def llm_ans_3_5(query):
    try:
      llm_response = qa_chain_3_5.invoke(query)
      ans = llm_response['result']
    except Exception as e:
        # Handle any errors and return the error message
        error_message = f"An error occurred: {str(e)}"
        return error_message,

    return ans
prompt_template2 = """
Please use only the given piece of context below and do not answer from your own knowledge.
Gave the answer in proper complete sentence.
{context}

The Question is asked below:
{question}
"""
replicate = Client(api_token="r8_5lkUYO901caPp57FpzJULDuTGOcLxi13gQ4zA")

PROMPT2 = PromptTemplate(
    template = prompt_template2,
    input_variables = ["context", "question"])

def Falcon_pipeline(question):
    documents = vectordb.similarity_search(question)
    
    context_list = [doc.page_content for doc in documents[:1]]
    combined_context = "\n".join(context_list)
    prompt = PROMPT2.format(context=combined_context, question=question)
    falcon_response=replicate.run(
    "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
    input={
        "prompt": prompt,
        "temperature": 0.7,
        'max_new_tokens': 2048,
        },
    )
    suggestions = ''.join([str(s) for s in falcon_response])

    return suggestions

def Llama_pipeline(question):
    documents = vectordb.similarity_search(question)
    
    context_list = [doc.page_content for doc in documents[:1]]
    combined_context = "\n".join(context_list)
    prompt = PROMPT2.format(context=combined_context, question=question)
    llama_response=replicate.run(
        "meta/llama-2-70b-chat",
        input={
            "prompt": prompt,
            "temperature": 0.75,
            'max_new_tokens': 2048,
        },
    )
    suggestions = ''.join([str(s) for s in llama_response])

    return suggestions
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
def calculate_rouge1_score(reference, candidate):
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    gpt_4_response = llm_ans(user_query)
    gpt_3_5_response = llm_ans_3_5(user_query)
    llama_response = Llama_pipeline(user_query)
    falcon_response = Falcon_pipeline(user_query)
    vectordb = FAISS.load_local(
    RAG.Persist_directory + '/faiss_index_hp',
    embeddings,
    allow_dangerous_deserialization=True
    )
    documents = vectordb.similarity_search(user_query)
    context_list = [doc.page_content for doc in documents[:1]]
    combined_context = "\n".join(context_list)
    candidates=[gpt_4_response,gpt_3_5_response,falcon_response,llama_response]
    models=['GPT-3.5_turbo','GPT-4','Falcon-40b','Llama-2-70b'] 
    scores = [calculate_rouge1_score(combined_context, candidate) for candidate in candidates]
    print(scores)
    max_score_index = scores.index(max(scores))

    responses = {
        "GPT-4": gpt_4_response,
        "GPT-3.5": gpt_3_5_response,
        "LLaMA": llama_response,
        "Falcon": falcon_response,
        "Best Model":models[max_score_index]
    }
    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True)