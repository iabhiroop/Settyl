import re
def pre_process(text):
    if text in ['PORT OUT', 'TERMINAL IN', 'PORT IN', 'DEPARTCU', 'TOLL PLAZA CROSSED', 'CFS OUT', 'CFS IN', 'TERMINAL OUT', 'GATE OUT']:
        return text
    else:
        return re.sub(r'\b(?:[A-Z]+|\d+\w*|\w*\d+)\b\s*|(\s*\(.*?\)\s*)|[^\w\s]', '', text).strip()