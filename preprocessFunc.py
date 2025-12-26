import pymorphy3 as pm3
import re

def string_preparation(string:str) -> str:
    mask = r'[^\w\s]'
    
    sub = re.sub(mask, '', string.lower())
    
    return sub


def normalize_string(string: str) -> str:
    morph = pm3.MorphAnalyzer()
    test_str = string_preparation(string=string)

    string = ''
    words = test_str.split()

    for i in words:
        s = morph.parse(i)
        
        string += s[0].normal_form + ' '
        
    return string.strip()
        