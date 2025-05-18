from finalModelWithAllFunc import agent_executor, useAgent

if __name__ == "__main__":
    while True:
        user_input = input()
        if user_input.lower() in ["exit", "quit", "goodbye"]:
            print("Chatbot: Goodbye!")
            break
        else:
            res = useAgent(agent_executor, user_input)
            print(res) 
            print("="*50)