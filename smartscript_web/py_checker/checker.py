import time
import pylint.epylint as linter
import tempfile
import logging
import regex as re
import os
import subprocess
import random
logger = logging.getLogger(__name__)



def parse(issues_str):
    if len(issues_str) == 0:
        return []
    items = issues_str.split('\n')[1:-1]
    issues = []
    for item in items:
        item_stripped = item.strip()
        if len(item_stripped) == 0:
            break
        m = re.match(r'(?:\S+:)(?P<lineno>[0-9]+):\s+(?P<type>[a-z]+)\s+\((?P<code>\S+),\s+(?:\S+),\s+(?P<extra>\S*)\)\s+(?P<desc>.*)', item_stripped)
        issues.append((m.group('type'), m.group('code'),
                       m.group('lineno'), m.group('desc') +
                       (': ' + m.group('extra') if len(m.group('extra')) > 0 else '')))
    return issues


# def check_code(code: str):
#     with tempfile.NamedTemporaryFile(mode='w') as code_file:
#         code_file.write(code)
#         code_file.flush()
#         (pylint_stdout, pylint_stderr) = linter.py_run(code_file.name, return_std=True)
#         issues_str = pylint_stdout.read()
#         logger.debug(issues_str)
#         return parse(issues_str)

def smart_script(code:str):
    time.sleep(0.2+2*random.random())
    issues_str = ""
    if code.startswith("\"\"\" format string bug"):
        issues_str="SmartScript AI has found 2 flaws in this code:\n1. Line 13: log(sys.stdout, _(\"Is it a bug? {result}\"))\nThe 'result' variable has no input.\n2. Line 15: '%(file)s: has %(bug}s'\nParentheses mismatch at \"%(bug}s\". "
            
    if code.startswith("\"\"\" Inconsistent data dependency"):
        issues_str="SmartScript AI has found 1 flaw in this code:\n1. Line 28: print(axis2)\nThe access of  variable 'axis2' might need to be under \"if with_components:\"."
    if code.startswith("{"):
        issues_str="SmartScript AI has found 1 flaw in this code:\n" \
            "1. Line 14: \"name\": \"400 Bizzell St, College Station, TX 77843\" \n" \
            "The \'name\' has an inconsistent name with its value assigned."
    if code.startswith("apiVersion"):
        issues_str="SmartScript AI has found 1 flaw in this code:\n" \
            "1. Line 8: missing \"matchLabels\"."

    return issues_str
def add_html_tags(issues_str, file_name):
    issues_str = issues_str[issues_str.index('\n')+1:]  # remove the first line
    pat = r'(?:{}):(?P<lineno>[0-9]+):'.format(file_name)
    ms = re.findall(pat, issues_str, re.MULTILINE)
    for m in ms:
        issues_str = re.sub(pat, '<a class="lineno" href="#">Line: {}</a>.'.format(m), issues_str, 1, re.MULTILINE)
    return issues_str
def add_html_tagsD(issues_str, file_name):
    pat = r'(?:{}):(?P<lineno>[0-9]+):'.format(file_name)
    ms = re.findall(pat, issues_str, re.MULTILINE)
    for m in ms:
        issues_str = re.sub(pat, '<a class="lineno" href="#">Line: {}</a>.'.format(m), issues_str, 1, re.MULTILINE)
    return issues_str
def run_code(code:str,ctype:str):
    return "For Security Reasons, this feature has been disabled."
    issue = ["import os","os.system","subprocess","os.popen"]
    # Security check

    for line in code:
        for iss in issue:
            if iss in line:
                return add_html_tagsD("This code cannot be executed due to security reasons."+"\n"+"Process finished at " + time.ctime())

    with tempfile.NamedTemporaryFile(mode='w',delete=False) as code_file:
        code_file.write(code)
        code_file.flush()
        code_file.close() # https://blog.csdn.net/weixin_44520259/article/details/89457875
        s = subprocess.Popen("python3 "+code_file.name, bufsize=0, stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                universal_newlines=True,shell=True)
        cmd_out = s.stdout.read()
        s.stdout.close()
        cmd_error = s.stderr.read()
        s.stderr.close()
        cmd_out = cmd_out
        cname = code_file.name
        cmd_out = cmd_out.replace(cname,ctype)
        cmd_error = cmd_error.replace(cname,ctype)
       
        code_file.close()
        os.remove(code_file.name)
        # print("================================")
        # print(cmd_out+cmd_error,code_file.name)
        # print("================================")
        # print("================================")
        # print(add_html_tagsD(cmd_out+cmd_error, code_file.name))
        # print("================================")
        return add_html_tagsD(cmd_out+"\n"+cmd_error+"\n"+"Process finished at " + time.ctime(), code_file.name)


def check_code(code: str):
    with tempfile.NamedTemporaryFile(mode='w',delete=False) as code_file:
        code_file.write(code)
        code_file.flush()
        code_file.close() # https://blog.csdn.net/weixin_44520259/article/details/89457875
        (pylint_stdout, pylint_stderr) = linter.py_run(code_file.name, return_std=True)
        issues_str = pylint_stdout.read()
        logger.debug(issues_str)
        code_file.close()
        os.remove(code_file.name)
        return add_html_tags(issues_str, code_file.name)
