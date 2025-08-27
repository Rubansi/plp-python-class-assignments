def modify_content(content):
    # Example modification: convert to uppercase
    return content.upper()

try:
    with open("input.txt", "r") as infile:
        original = infile.read()
        modified = modify_content(original)

    with open("output.txt", "w") as outfile:
        outfile.write(modified)

    print("File modified and saved as output.txt")

except FileNotFoundError:
    print("Error: input.txt not found.")
except Exception as e:
    print(f"Unexpected error: {e}")
