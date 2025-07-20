1. it returns the NULL character
2. `__repr__` returns chars string representation which is unambiguous for debugging purpose, whilst `__str__` returns
   human-readable format which can be ambiguous sometimes.
3. it prints out empty for the first line, for second print it shows `this is a teststring` because NULL is not a space character.


1. UTF-16 and UTF-32 has longer encoded bytes compared to UTF-8 encoding for the same string.
2. multiple bytes can be decoded on one character, but the wrong solution decodes only one byte at a time
3. `b = bytes([255, 255])`
