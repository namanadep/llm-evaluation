import time
from openai import OpenAI
import tiktoken

class ContextWindowTester:
    def __init__(self):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="dummy"
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        return len(self.encoding.encode(text))
    
    def create_test_text(self, target_tokens):
        # Each base sentence is about 10 tokens
        base_sentence = "This is a simple test sentence for context window. "
        markers = {
            "START": "UNIQUE_START_MARKER_123",
            "MIDDLE": "UNIQUE_MIDDLE_MARKER_456",
            "END": "UNIQUE_END_MARKER_789"
        }
        
        # Calculate how many sentences we need
        tokens_per_sentence = self.count_tokens(base_sentence)
        sentences_needed = (target_tokens - 100) // (tokens_per_sentence * 2)  # Divide by 2 as we'll use it twice
        
        filler = base_sentence * sentences_needed
        
        test_text = f"{markers['START']}\n{filler}\n{markers['MIDDLE']}\n{filler}\n{markers['END']}"
        actual_tokens = self.count_tokens(test_text)
        
        print(f"Created test text with {actual_tokens:,} tokens")
        return test_text, actual_tokens

    def run_test(self, target_tokens):
        print(f"\nStarting context window test for {target_tokens:,} tokens...")
        
        # Create test text
        start_time_total = time.time()
        print("Generating test text...")
        test_text, actual_tokens = self.create_test_text(target_tokens)
        
        # Run the test
        print("Sending API request...")
        query = "Find and list all the UNIQUE markers in the text (START, MIDDLE, and END markers)"
        
        start_time_api = time.time()
        try:
            response = self.client.chat.completions.create(
                model="qwen2.5",
                messages=[
                    {"role": "user", "content": test_text},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=100
            )
            end_time = time.time()
            
            # Calculate timings
            api_time = end_time - start_time_api
            total_time = end_time - start_time_total
            
            # Print results
            print("\nTest Results:")
            print(f"Tokens tested: {actual_tokens:,}")
            print(f"API response time: {api_time:.2f} seconds")
            print(f"Total test time: {total_time:.2f} seconds")
            print(f"Tokens per second: {actual_tokens/api_time:,.2f}")
            print("\nModel Response:")
            print(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error during test: {str(e)}")

def main():
    tester = ContextWindowTester()
    
    # Menu for selecting context window size
    sizes = {
        '1': 1_000,
        '2': 2_000,
        '3': 4_000,
        '4': 8_000,
        '5': 16_000,
        '6': 32_000,
        'c': None  # Custom size
    }
    
    while True:
        print("\nSelect context window size to test:")
        print("1. 1K tokens")
        print("2. 2K tokens")
        print("3. 4K tokens")
        print("4. 8K tokens")
        print("5. 16K tokens")
        print("6. 32K tokens")
        print("c. Custom size")
        print("q. Quit")
        
        choice = input("Enter your choice: ").lower()
        
        if choice == 'q':
            break
        elif choice == 'c':
            try:
                custom_size = int(input("Enter desired token count: "))
                tester.run_test(custom_size)
            except ValueError:
                print("Please enter a valid number")
        elif choice in sizes:
            tester.run_test(sizes[choice])
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
