from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template
template = """
Answer the question below.
Here is the conversation history: {history}
Question: {question}
Answer:
"""
# Initialize the model and chain
model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def handle_conversations():
    history = ""  # Initialize conversation history
    print("Ask me anything! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")  # Get user input
        if user_input.lower() == "exit":  # Exit condition
            break

        # Invoke the chain with history and the user question
        result = chain.invoke({"history": history, "question": user_input})

        # Safely handle the result
        response = result.text if hasattr(result, "text") else str(result)
        print("Bot:", response)

        # Update the conversation history
        history += f"\nUser: {user_input}\nBot: {response}"


if __name__ == "__main__":
    handle_conversations()
