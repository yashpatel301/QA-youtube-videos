from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st

def convert_question(inputs):
    if inputs['query_lan'] != inputs['video_lan']:
        return GoogleTranslator(source=inputs['query_lan'], target=inputs['video_lan']).translate(inputs['query'])
    return inputs['query']


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_inputs():
    # video_id = input("Enter video id: ")
    # video_lan = input("Enter video language: ")
    video_id = st.text_input("Enter video id: ", key="video_id_input")
    video_lan = st.radio("Select transcript language:", options=["en", "es", "hi", "fr", "de", "it", "pt", "ru"], key="video_lan_input")
    # translator = GoogleTranslator(source='auto', target='en')
    # languages = translator.get_supported_languages(as_dict=True)
    # while video_lan not in list(languages.keys()):
    #     print("Language not supported")
    #     video_lan = input("Enter video language: ")
    # video_lan = languages[video_lan]
    query = st.text_input("Enter query: ", key="query_input")
    query_lan = st.radio("Select query language:", options=["en", "es", "hi", "fr", "de", "it", "pt", "ru"], key="query_lan_input")
    # while query_lan not in list(languages.keys()):
    #     print("Language not supported")
    #     query_lan = input("Enter your query language: ")
    # query_lan = languages[query_lan]
    return {'video_lan': video_lan, 'query_lan': query_lan, 'query': query}, video_id


def get_embeddings(video_id, transcript, embeddings):
    if os.path.exists(f"faiss_index/{video_id}/"):
        vector_store = FAISS.load_local(f"faiss_index/{video_id}", embeddings, allow_dangerous_deserialization=True)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_doc = " ".join([entry["text"] for entry in transcript])
        chunks = splitter.split_text(final_doc)

        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(f"faiss_index/{video_id}")
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


if __name__ == '__main__':
    load_dotenv()
    parser = StrOutputParser()
    model = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", max_tokens=500)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    
    st.title("YouTube Video Q&A")
    st.sidebar.subheader("This app allows you to ask questions about a YouTube video transcript.")
    st.sidebar.subheader("It uses OpenAI's GPT-3.5-turbo model to generate answers based on the transcript.")

    st.subheader("Enter the video ID and select the language of the video transcript.")
    inputs, video_id = get_inputs()
        
    if st.button("Get Transcript", key="get_transcript_button"):
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[inputs['video_lan']])
        retriever = get_embeddings(video_id, transcript, embeddings)

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""
                You are a helpful assistant that answers questions about a YouTube video transcript.
                {context}
                Question: {query}
                Answer in the same language as the question.
                If you don't know the answer, say "I don't know"."""
            )
            
        parellel_chain = RunnableParallel({
            'context': RunnableLambda(convert_question) | retriever | RunnableLambda(format_docs),
            'query': RunnableLambda(convert_question)
        })

        chain = parellel_chain | prompt | model | parser
        response = chain.invoke(inputs)
        final_resp = GoogleTranslator(source=inputs['video_lan'], target='en').translate(response)
        st.success("Answer: ")
        st.write(final_resp)