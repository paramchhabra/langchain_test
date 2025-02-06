# import streamlit as st
# from langchain_experimental.agents import create_csv_agent
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from dotenv import load_dotenv
# from transformers import pipeline,AutoTokenizer
# import os



# def main():
#     load_dotenv()

#     st.set_page_config(page_title="Ask CSV")
#     st.header("Ask CSV")

#     user_csv = st.file_uploader("Upload Your CSV", type="csv")

#     if user_csv is not None:
#         user_ques = st.text_input("Ask a Question about the CSV")
#         # user_ques.

#         if user_ques is not None and user_ques != "":
            
#             model_name = "Wimflorijn/t5-text2text"
#             tokenizer = AutoTokenizer.from_pretrained(model_name)

#             inputs = tokenizer(user_ques, return_tensors="pt")
#             input_ids = inputs["input_ids"]

#             # Wrap the pipeline with LangChain's HuggingFacePipeline
#             llm = HuggingFacePipeline.from_model_id(                
#                 model_id=model_name,
#                 tokenizer=tokenizer,
#                 input_ids=input_ids,
#                 task="text2text-generation",  
#                 )
            
#             agent = create_csv_agent(llm,user_csv, verbose=True,allow_dangerous_code=True)
#             # st.write(f"Your Question was : {user_ques}")
#             response = agent.run(user_ques)
#             st.write(response)

    

# if __name__=="__main__":
#     main()


from langchain_experimental.agents import create_csv_agent
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

with open("data.csv","r") as f:
    model_name = "google/flan-t5-base"  # Replace with the model of your choice
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Step 2: Create Hugging Face Pipeline
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    ques = "How many rows are present in the file?"

    agent = create_csv_agent(llm,f,verbose=True,allow_dangerous_code=True)
    ans = agent.run(ques)
    print(str(ans))