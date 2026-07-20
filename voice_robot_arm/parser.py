COLORS = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["cube", "sphere", "cylinder"]

def parse_command(text):
    text = text.lower()

    color = None
    shape = None

    for c in COLORS:
        if c in text:
            color = c

    for s in SHAPES:
        if s in text:
            shape = s

    return color, shape