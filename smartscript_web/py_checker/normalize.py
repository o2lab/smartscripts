# encoding=utf-8
"""
In some cases, we might want to normalize the code,
e.g., a.b(c) => CLASS_INSTANCE . CALL ( ARGUMENT )
"""
from string import Formatter
from redbaron import RedBaron, NameNode, GetitemNode, CallNode, DotNode
from baron import BaronError
import regex as re


def get_format_string_keywords(stmt: str):
    """
    https://stackoverflow.com/questions/25996937/how-can-i-extract-keywords-from-a-python-format-string
    :param stmt:
    :return:
    """
    fieldnames = [fname for _, fname, _, _ in Formatter().parse(stmt) if fname]
    return fieldnames


def normalize_format_string(stmt: str):
    """
    Only support python3 style format string so far.
    How to check python2?
    _(..., log(...), ...)
    :param stmt:
    :return:
    """
    if re.search(r'(\{[\S ]*?\})', stmt) is None:
        return stmt
    try:
        red = RedBaron(stmt)
    except (BaronError, AssertionError) as err:
        return None
    name_nodes = red.find_all('name')
    renaming_mapping = {}
    cnt = 0
    for name_node in name_nodes:
        if isinstance(name_node.next, GetitemNode) or isinstance(name_node.next, DotNode):
            if name_node.value in renaming_mapping:
                name_node.value = renaming_mapping[name_node.value]
            else:
                renaming_mapping[name_node.value] = 'INSTANCE{}'.format(cnt)
                name_node.value = renaming_mapping[name_node.value]
                cnt += 1
        elif isinstance(name_node.next, CallNode):
            if name_node.value in renaming_mapping:
                name_node.value = renaming_mapping[name_node.value]
            else:
                renaming_mapping[name_node.value] = 'FUNC{}'.format(cnt)
                name_node.value = renaming_mapping[name_node.value]
                cnt += 1
        else:
            if name_node.value in renaming_mapping:
                name_node.value = renaming_mapping[name_node.value]
            elif name_node.value.isdigit():  # skip constant number
                continue
            else:
                renaming_mapping[name_node.value] = 'VAR{}'.format(cnt)
                name_node.value = renaming_mapping[name_node.value]
                cnt += 1
    string_nodes = red.find_all('string')
    for string_node in string_nodes:
        matches = re.findall(r'(\{[\S ]*?\})', string_node.value)
        if matches is None or len(matches) == 0:
            continue
        # new_val = string_node.value[0] + ''.join(matches) + string_node.value[-1]
        new_val = '"' + ' '.join(matches) + '"'
        # for old_id, new_id in renaming_mapping.items():
        #     # for now just use replace
        #     # Maybe we should find a way to get all identifiers in a format string later
        #     new_val = new_val.replace(old_id, new_id)
        keywords = get_format_string_keywords(string_node.value)
        for keyword in keywords:
            if keyword in renaming_mapping:
                new_val = new_val.replace(keyword, renaming_mapping[keyword])
            else:
                renaming_mapping[keyword] = 'VAR{}'.format(cnt)
                new_val = new_val.replace(keyword, renaming_mapping[keyword])
                cnt += 1
        string_node.value = new_val
    return red.dumps()


def normalize_dict_key(method: str) -> str:
    """
    Remove statements unrelated to dict key access or field access
    :param method:
    :return:
    """
    red = RedBaron(method)
    # remain_stmts = []
    method_red = red.find('def')
    for i, stmt in enumerate(method_red.value):
        if len(stmt.find_all('dictnode')) != 0 or len(stmt.find_all('getitem')) != 0 or len(stmt.find_all('dotnode')) != 0:
            # remain_stmts.append(stmt.dumps())
            continue
        else:
            del method_red.value[i]
    # red.value = '\n'.join(remain_stmts)
    return red.dumps()
