import os
import sys
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# ------------------------------------------------------------------
# 1. Define the Agent State (Now with Memory)
# ------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    # Using Annotated with operator.add tells LangGraph to append to this list
    # rather than overwriting it, creating a persistent memory log.
    chat_history: Annotated[list, operator.add] 
    context: str
    intent: str  
    generation: str

class ETLAssistantAgent:
    def __init__(self, docs_path="etl_docs.txt"):
        self.docs_path = docs_path
        self.vectorstore = None
        self.llm = None
        self.app = None
        self.setup_environment()
        self.build_graph()

    def setup_environment(self):
        """Initializes the LLM and the Vector Store."""
        if not os.path.exists(self.docs_path):
            print(f"Error: '{self.docs_path}' not found. Please create it.")
            sys.exit(1)
            
        if "GROQ_API_KEY" not in os.environ:
            print("Error: GROQ_API_KEY environment variable not found.")
            sys.exit(1)

        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        loader = TextLoader(self.docs_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(splits, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

    # ------------------------------------------------------------------
    # 2. Define Graph Nodes
    # ------------------------------------------------------------------
    def retrieve_node(self, state: AgentState):
        """Retrieves context based on the question."""
        docs = self.retriever.invoke(state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    def classify_intent_node(self, state: AgentState):
        """Classifies intent."""
        prompt = ChatPromptTemplate.from_template(
            "Analyze the user query and classify the intent into one of two categories: 'config' or 'qa'.\n"
            "Return ONLY the word 'config' if the user is asking to generate, validate, or modify a JSON configuration.\n"
            "Return ONLY the word 'qa' if the user is asking a general question.\n"
            "Query: {question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"question": state["question"]}).strip().lower()
        if "config" not in intent:
            intent = "qa"
        return {"intent": intent}

    def qa_generator_node(self, state: AgentState):
        """Handles general conceptual questions with conversational memory."""
        # Format chat history for the prompt
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.get("chat_history", [])])
        
        prompt = ChatPromptTemplate.from_template(
            "You are an ETL Assistant. Answer the question based STRICTLY on the context provided.\n"
            "Previous Conversation:\n{history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "history": history_str, 
            "context": state["context"], 
            "question": state["question"]
        })
        
        # Append the current interaction to the memory log
        new_memory = [
            {"role": "User", "content": state["question"]},
            {"role": "Assistant", "content": response}
        ]
        return {"generation": response, "chat_history": new_memory}

    def config_specialist_node(self, state: AgentState):
        """Handles JSON configuration generation and validation with memory."""
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.get("chat_history", [])])
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert ETL Configuration Validator. Use the context and previous conversation to handle the user's request.\n"
            "If the user is asking to modify a previously generated config, use the Previous Conversation to understand what to change.\n"
            "If asked to VALIDATE: Evaluate internally. Respond ONLY with 'Valid.' or 'Invalid.' followed by a brief, 1-sentence explanation. Do NOT list out every rule check.\n"
            "If asked to GENERATE: Provide only valid JSON matching the schema.\n\n"
            "Previous Conversation:\n{history}\n\n"
            "Context:\n{context}\n\nRequest: {question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "history": history_str, 
            "context": state["context"], 
            "question": state["question"]
        })
        
        new_memory = [
            {"role": "User", "content": state["question"]},
            {"role": "Assistant", "content": response}
        ]
        return {"generation": response, "chat_history": new_memory}

    # ------------------------------------------------------------------
    # 3. Define Routing Logic & Compile Graph
    # ------------------------------------------------------------------
    def route_task(self, state: AgentState):
        return state["intent"]

    def build_graph(self):
        """Constructs the LangGraph state machine with memory."""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("classify", self.classify_intent_node)
        workflow.add_node("qa", self.qa_generator_node)
        workflow.add_node("config", self.config_specialist_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "classify")
        
        workflow.add_conditional_edges("classify", self.route_task, {"qa": "qa", "config": "config"})

        workflow.add_edge("qa", END)
        workflow.add_edge("config", END)

        # Initialize the checkpointer for thread-level memory
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def ask(self, query: str, thread_id: str = "demo_thread_1"):
        """Executes the graph using a specific thread ID for memory persistence."""
        inputs = {"question": query}
        # The configurable dictionary passes the thread_id to the checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        
        final_generation = ""
        
        print("\n--- Execution Trace ---")
        for output in self.app.stream(inputs, config=config):
            for node_name, state_update in output.items():
                print(f"✅ Node Executed: [{node_name}]")
                if "generation" in state_update:
                    final_generation = state_update["generation"]
        print("-----------------------\n")
        
        return final_generation

def main():
    print("Initializing Agentic ETL Assistant with Conversational Memory...")
    assistant = ETLAssistantAgent()
    
    # We use a static thread ID for the CLI session so it remembers the whole conversation
    session_thread_id = "interview_demo_1"
    
    print("\n==================================================")
    print("ETL Operator Configuration Assistant is ready!")
    print("Type 'exit' to stop.")
    print("==================================================\n")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            print("Assistant: Processing workflow...")
            response = assistant.ask(user_input, thread_id=session_thread_id)
            print(f"\nAssistant:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()