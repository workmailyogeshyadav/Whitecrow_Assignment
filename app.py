"""Small interactive CLI demonstrating multi-turn conversational RAG."""
from src.graph import ask

def main():
    print("\nNIST Agentic RAG Assistant")
    print("Ask questions about the NIST AI RMF. Type 'exit' to stop.\n")
    history: list[dict[str, str]] = []

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = ask(question, history)
            answer = result.get("answer", "I could not generate an answer.")
            print(f"\nAssistant: {answer}")

            sources = result.get("sources", [])
            if sources:
                print("\nSources:")
                for source in sources:
                    print(
                        f"- {source['document']} | "
                        f"Page {source['page']} | {source['section']}"
                    )
            print()

            # Keep only a short rolling history to control prompt size.
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            history = history[-10:]
        except Exception as exc:
            print(f"\nAn error occurred: {exc}\n")

if __name__ == "__main__":
    main()
