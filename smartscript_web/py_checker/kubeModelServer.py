#!/home/wangluochao/anaconda3/envs/crawler/bin/python

from symspellpy import SymSpell, Verbosity
import pandas as pd
import pickle, sys, yaml
import re
from collections import defaultdict
from termcolor import colored, cprint
import re


def check_typo(sym_spell, line_words):
    """check spelling type

    Args:
        sym_spell (symspellpy.SymSpell): a SymSpell instance to check typo
        line_words (list): a list of (line_number, word) for typo-checking
    """
    typo_message = []
    for n, word in line_words:
        suggestions = sym_spell.lookup(word,
                                       Verbosity.CLOSEST,
                                       max_edit_distance=2,
                                       transfer_casing=True)
        if suggestions:
            suggestion = suggestions[0]
            suggested_word = suggestion.term
            if word.lower() != suggested_word:
                typo_message.append(
                    f'ypo Warning at line {n} for word "{word}"; suggestion is "{suggested_word}"'
                )
    return typo_message


def check_missing_entry(association_rules, entries):
    """check missing entry errors using the rules extracted with association rules mining.
    
    Args:
        rule (dict): association of always happen together pairs. association_rules[antecedent] = consequent
        entries (set): entries to check
    """
    missing_messages = []
    keys = set(association_rules.keys())
    for ante in entries:
        if ante in keys:
            for conse in association_rules[ante]:
                if conse not in entries:
                    missing_messages.append(
                        f'Missing Entry Warning: expect "{conse}" when "{ante}" presents'
                    )
    return missing_messages


def check_incorrect_type(type_rules, entry_type):
    """check whether entry_type follow type_rules
    
    Args:
        type_rules (dict): dict of str to set
        entry_type (list): list of tuple(entry, type)
    """
    incorrect_type_messages = []
    keys = set(type_rules.keys())
    for entry, typ in entry_type:
        if entry in keys:
            if typ not in type_rules[entry]:
                pattern = r"<|>|class|\""
                if len(type_rules[entry]) > 1:
                    incorrect_type_messages.append(
                        f'Incorrect Type Warning: expect one of {re.sub(pattern, "", str(type_rules[entry]))}, but got {re.sub(pattern, "", str(typ))} for "{entry}"'
                    )
                else:
                    incorrect_type_messages.append(
                        f'Incorrect Type Warning: expect {re.sub(pattern, "", str(list(type_rules[entry])[0]))}, but got {re.sub(pattern, "", str(typ))} for "{entry}"'
                    )
    return incorrect_type_messages


def flatten_dict(flat_dict, pre_key, d):
    """dfs
    
    Args:
        flat_dict (list): list of dict
        pre_key (str): prefix of key
        d (dict): [description]
    """
    for k in d.keys():
        new_pre_key = f"{pre_key}/{str(k)}"
        flat_dict[new_pre_key] = d[k]
        if isinstance(d[k], dict):
            flatten_dict(flat_dict, new_pre_key, d[k])


def generate_flat_dict(yaml_content):
    try:
        docs = yaml.load_all(yaml_content, Loader=yaml.FullLoader)
    except:
        raise SyntaxError("I have raised an Exception")
    docs_flat_dict = []
    apiVersion_kind_missing_messages = []
    for doc in docs:
        if not doc:
            apiVersion_kind_missing_messages.append([])
            continue
        keys = doc.keys()
        error_message = []
        if 'apiVersion' not in keys or 'kind' not in keys:
            if 'apiVersion' not in keys:
                error_message.append(
                    f'Missing Entry Warning: expect "apiVersion" presented in the this document'
                )
            if 'kind' not in keys:
                error_message.append(
                    f'Missing Entry Warning: expect "kind" presented in this document'
                )
            apiVersion_kind_missing_messages.append(error_message)
            docs_flat_dict.append(dict())
        else:
            apiVersion_kind_missing_messages.append([])
            flat_dict = dict()
            flat_dict['apiVersion'] = doc['apiVersion']
            flat_dict['kind'] = doc['kind']
            flat_dict[f"apiVersion({doc['apiVersion']})"] = doc['apiVersion']
            flat_dict[f"kind({doc['kind']})"] = doc['kind']
            pre_key = f"apiVersion({doc['apiVersion']})/kind({doc['kind']})"
            doc.pop('apiVersion')
            doc.pop('kind')
            flatten_dict(flat_dict, pre_key, doc)
            docs_flat_dict.append(flat_dict)
    return docs_flat_dict, apiVersion_kind_missing_messages


def generate_entry_type(docs_flat_dict):
    """[summsary]
    
    Args:
        docs_flat_dict ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    docs_entry_type = []
    for flat_dict in docs_flat_dict:
        entry_type = []
        for entry in flat_dict:
            entry_type.append((entry, type(flat_dict[entry])))
        docs_entry_type.append(entry_type)
    return docs_entry_type


def parse_words(text):
    """Create a non-unique wordlist from sample text
    language independent (e.g. works with Chinese characters)
    """
    # // \w Alphanumeric characters (including non-latin
    # characters, umlaut characters and digits) plus "_". [^\W_] is
    # the equivalent of \w excluding "_".
    # Compatible with non-latin characters, does not split words at
    # apostrophes.
    # Uses capturing groups to combine a negated set with a
    # character set
    matches = re.findall(r"(([^\W_]|['’])+)", text)
    # The above regex returns ("ghi'jkl", "l") for "ghi'jkl", so we
    # extract the first element
    matches = [match[0] for match in matches]
    return matches


def get_results(yaml_content,
                rules_path,
                type_path,
                sym_spell_path=None,
                out_file=None):
    line_word = []
    lines = yaml_content.split('\n')
    # ignore empty lines and comment lines at the beginning
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        else:
            break
    # the first document does not need to startw with '---'
    if lines[i].strip() == '---':
        i += 1
    docs_line_number = [i + 1]
    start_line_index = i + 1
    for j, line in enumerate(lines[start_line_index:]):
        line_index = start_line_index + j
        line = line.strip()
        if line.startswith('#'):
            continue
        if line == '---':
            docs_line_number.append(line_index + 1)
        words = parse_words(line)
        for word in words:
            line_word.append((line_index + 1, word))

    # if there is still codes after the last '---', the rest of code is
    # still consider one valid document.
    for i in range(docs_line_number[-1], len(lines)):
        line = lines[i].strip()
        if (not line.startswith('#')) and (line != ''):
            docs_line_number.append(len(lines))
            break
    # check typo
    # sym_spell = SymSpell()
    # sym_spell.load_pickle(sym_spell_path)
    # typo_message = check_typo(sym_spell, line_word)
    # if typo_message:
    #     for m in typo_message:
    #         print(m)
    #         if out:
    #             out.write(m+'\n')
    #     print()

    # load rules
    df = pd.read_csv(rules_path)
    association_rules = defaultdict(set)
    for index, row in df.iterrows():
        association_rules[row['antecedents']].add(row['consequents'])

    with open(type_path, 'rb') as f:
        type_rules = pickle.load(f)
    docs_flat_dict, apiVersion_kind_missing_messages = generate_flat_dict(
        yaml_content)
    docs_entry_type = generate_entry_type(docs_flat_dict)

    # check missing entry and incorrect type error
    all_messages = ''
    for i in range(len(docs_line_number) - 1):
        flat_dict = docs_flat_dict[i]
        entry_type = docs_entry_type[i]
        missing_messages = check_missing_entry(association_rules,
                                               set(flat_dict.keys()))
        incorrect_type_messages = check_incorrect_type(type_rules, entry_type)
        if missing_messages or incorrect_type_messages or apiVersion_kind_missing_messages[
                i]:
            warning = 'Warnings'
            # warning = colored('Warnings', 'red')
            m = warning + f' for Document Starting from line {docs_line_number[i]} to {docs_line_number[i+1]}:'
            all_messages = all_messages + m + '\n'
            # print(m)
            for m in incorrect_type_messages + missing_messages + apiVersion_kind_missing_messages[
                    i]:
                # print(m)
                all_messages = all_messages + m + '\n'
                # if out:
                #     out.write(m+'\n')
    if out_file is not None:
        with open(out_file, 'w', encoding='utf-8') as out:
            out.write(all_messages)
    return all_messages


def main(yaml_content):
    rules_path = "/home/smartscript/smartscript_web/py_checker/confidence1_support002_rules.csv"
    type_path = "/home/smartscript/smartscript_web/py_checker/entry_type.pkl.txt"
    all_messages = get_results(yaml_content, rules_path, type_path)
    all_messages = re.sub(r'(warnings|warning)',
                          r'<font color="yellow">\1</font>',
                          all_messages,
                          flags=re.IGNORECASE)
    return all_messages + '<font color="#00bfff">Finish!</font>\n'


# if __name__ == "__main__":
#     yaml_file_path = sys.argv[1]
#     sym_spell_path = "/home/wangluochao/smartscript/configuration/sym_spell_yaml.pkl"
#     rules_path = "/home/wangluochao/smartscript/configuration/confidence1_support002_rules.csv"
#     type_path = "/home/wangluochao/smartscript/configuration/entry_type.pkl"
#     main(yaml_file_path, sym_spell_path, rules_path, type_path)
