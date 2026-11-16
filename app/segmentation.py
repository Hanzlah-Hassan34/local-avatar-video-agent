import re


def segment_script(script: str, max_characters: int = 300) -> list[str]:
    """Split a script at sentence boundaries, keeping each segment manageable."""
    if max_characters < 1:
        raise ValueError("max_characters must be greater than zero")

    sentences = re.findall(r".+?(?:[.!?]+(?=\s|$)|$)", script.strip(), flags=re.DOTALL)
    segments: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_characters:
            if current:
                segments.append(current)
                current = ""
            segments.extend(
                sentence[index : index + max_characters].strip()
                for index in range(0, len(sentence), max_characters)
            )
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_characters:
            segments.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        segments.append(current)

    return segments
