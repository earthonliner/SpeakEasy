import re
import subprocess


def remove_non_english_chars(text):
    cleaned_text = re.sub(r"[^a-zA-Z\s.,!?]'", '', text)
    return cleaned_text
  

def run_terminal_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        if output:
          print("Output:", output)
    except subprocess.CalledProcessError as e:
        print("Error:", e)


if __name__ == '__main__':
    # Example usage
    text = "Hello, 123! This is a sample text with non-English characters like é, ñ, and ç."
    cleaned_text = remove_non_english_chars(text)
    print(cleaned_text)