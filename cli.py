import os
import sys
from skb_graphrag import GraphRAG # Assumes your GraphRAG class is in backend.py

# Optional: Add colors for better readability in the terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # 1. Initialize
    print(f"{Colors.HEADER}🕸️  Initializing Neo4j GraphRAG CLI...{Colors.ENDC}")
    
    try:
        rag = GraphRAG()
        print(f"{Colors.GREEN}✅ Connected to Neo4j Database{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.YELLOW}❌ Connection Error: {e}{Colors.ENDC}")
        sys.exit(1)

    print(f"\n{Colors.BOLD}Instructions:{Colors.ENDC}")
    print(" - Type your question and press Enter.")
    print(" - Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    # 2. The Chat Loop
    while True:
        try:
            user_input = input(f"\n{Colors.BLUE}You > {Colors.ENDC}")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            print(f"{Colors.YELLOW}🕷️  Thinking... (Searching Knowledge Graph){Colors.ENDC}")

            # A. Retrieve & Display Context (The "Sidebar" feature)
            context_data = rag.retrieve_graph_context(user_input)
            
            if context_data:
                print(f"\n{Colors.BOLD}--- 🔍 RETRIEVED CONTEXT ---{Colors.ENDC}")
                formatted_context = rag.format_context(context_data)
                # Indent context for visual separation
                print("\n".join(f"  {line}" for line in formatted_context.splitlines()))
                print(f"{Colors.BOLD}-----------------------------{Colors.ENDC}\n")
                
                # B. Generate Answer
                answer = rag.generate_answer(user_input)
                print(f"{Colors.GREEN}🤖 AI > {answer}{Colors.ENDC}")
            else:
                print(f"{Colors.YELLOW}🤖 AI > I couldn't find any relevant nodes in the graph to answer that.{Colors.ENDC}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"{Colors.YELLOW}Error: {e}{Colors.ENDC}")

    rag.close()

if __name__ == "__main__":
    main()
