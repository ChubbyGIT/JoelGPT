from config import COLOR_USER

def get_multiline_input(prompt="You: "):
    print(COLOR_USER + prompt, end="", flush=True)
    lines = []
    while True:
        line = input()
        if line.endswith("/"):
            lines.append(line[:-1])
            print(COLOR_USER + "... ", end="", flush=True)
        else:
            lines.append(line)
            break
    return "\n".join(lines).strip()
