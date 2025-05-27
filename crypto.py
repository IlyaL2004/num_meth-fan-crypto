import math
import PyPDF2
import re

variant = 225
print(f"Номер варианта: {variant}")

numbers = []

try:
    with open('lab.pdf', 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""

        for page in pdf_reader.pages:
            full_text += page.extract_text()

        pattern = re.compile(
            r'(?:a|b)\[(\d+)\]\s*=\s*((?:\d|\s)+)',
            re.DOTALL
        )
        matches = pattern.findall(full_text)

        numbers_dict = {}
        for idx, num_str in matches:
            num = int(re.sub(r'\D', '', num_str))
            numbers_dict[int(idx)] = num

        max_idx = max(numbers_dict.keys(), default=-1)
        for idx in range(max_idx + 1):
            if idx in numbers_dict:
                numbers.append(numbers_dict[idx])

except FileNotFoundError:
    print("Ошибка: Файл 'lab.pdf' не найден!")
    exit(1)


n = numbers[variant]
print(f"n = {n}")

p, q = 0, 0
for num in numbers:
    if num == n or num == 0:
        continue
    gcd = math.gcd(n, num)
    if 1 < gcd < n:
        p = gcd
        q = n // p
        break

if p != 0:
    print("Найдены множители:")
    print(f"p = {p}")
    print(f"q = {q}")
    print(f"Проверка: {n == p * q}")
else:
    print("Делители не найдены")
