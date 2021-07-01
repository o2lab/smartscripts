# coding=utf-8
"""
Train subword units model using SentencePiece Library:
https://github.com/google/sentencepiece
By default, we use BPE algorithm.
"""
import os
import regex as re
import toml
import parso
import sentencepiece as spm

import logutils

# Default token used by sp
EOS = '</s>'
BOS = '<s>'
UNK = '<unk>'
PAD = '<pad>'

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train(corpus_files, model_prefix='spm', vocab_size=10000,
          model_type='bpe'):
    """
    Train the model.
    # "--pad_piece={} --unk_piece={} --bos_piece={} --eos_piece={}"

    Args:
        corpus_files: A list of input files (.txt).
        model_prefix: The model name.
        vocab_size: The number of tokens appear in the vocabulary.
        model_type: Supported types are: bpe, unigram, char and word.
                    We use bpe by default.
    """
    spm.SentencePieceTrainer.Train(
        "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        "--character_coverage=1.0 "
        "--input={} --model_prefix={} --vocab_size={} --model_type={}"
        .format(','.join(corpus_files), model_prefix, vocab_size, model_type))


def load_model(model_file):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)
    return sp


def encode(sp: spm.SentencePieceProcessor, sent):
    return sp.encode_as_ids(sent)


def decode(sp: spm.SentencePieceProcessor, ids):
    return sp.decode_ids(ids)


def preprocess(sent):
    # separate identifier and operators (e.g., +,-,*,/,(,...)
    pat = r"""([\(\)\.,;\\\/\{\}\[\]\"\'\+\-\*\|\^\?])"""
    subst = r""" \1 """
    sent = re.sub(pat, subst, sent)
    return sent


def tokenize_format_string(stmt: str, tokens: list):
    """
    stmt has form of "{} {arg_i: %2.f}".
    We need to take each symbol and variable name as a token
    :param stmt:
    :param tokens:
    :return:
    """
    tokens.append('"')
    fields = stmt[1:-1].split()
    for field in fields:
        tokens.append('{')
        if ':' in field:
            inner = field[1:-1].split(':')
            tokens.append(inner[0])  # var name
            tokens.append(':')
            tokens.append(inner[1])  # format control
        else:
            tokens.append(field[1:-1])
        tokens.append('}')
    tokens.append('"')


def traverse(node, tokens, handle_format_string=False):
    if isinstance(node, parso.tree.Leaf):
        # tokens.append(node.value)
        # code = node.get_code()  # get_code() return all original code including comments and whitespaces
        code = node.value
        if code is None or len(code) == 0:
            return
        if handle_format_string and re.search(r'(\{[\S ]*?\})', code) is not None:
            tokenize_format_string(code, tokens)
        else:
            tokens.append(code)
        tokens.append(' ')
        return
    for child in node.children:
        traverse(child, tokens)


def ast_tokenize(code_files):
    # Using AST to tokenize first to keep operators
    logger = logutils.get_logger(__name__)
    new_code_files = []
    for code_file in code_files:
        try:
            code = open(code_file).read()
            ast = parso.parse(code)
            tokens = []
            traverse(ast, tokens)
            new_code_file = os.path.join(os.path.dirname(code_file),
                                         os.path.basename(code_file).replace('.py', '.pyt').replace(' ', ''))
            new_code = ''.join(tokens)
            with open(new_code_file, 'w') as writer:
                writer.write(new_code)
            new_code_files.append(new_code_file)
        except OSError as err:
            logger.error('Fail to open code file: {}'.format(err))
        except UnicodeDecodeError as err:
            logger.error("Error when reading file: {}, {}".format(code_file, err))
    return new_code_files


# used for tokenize format string statement
def ast_tokenize_str(code: str, handle_format_string=False):
    try:
        ast = parso.parse(code)
    except (TypeError, KeyError) as err:
        return []
    tokens = []
    traverse(ast, tokens, handle_format_string)
    tokens = map(lambda t: t.strip(), tokens)
    tokens = list(filter(lambda t: t != ' ' and len(t) > 0, tokens))
    return tokens


def train_on_code_files(dataset_folder, vocab_size=10000):
    logger = logutils.get_logger(__name__)
    code_files = []
    for bug_folder in os.listdir(dataset_folder):
        if not os.path.isdir(os.path.join(dataset_folder, bug_folder)):
            continue
        before_folder = os.path.join(dataset_folder, bug_folder, 'before')
        for code_file in os.listdir(before_folder):
            if code_file.endswith('.py'):
                code_files.append(os.path.join(before_folder, code_file))
        after_folder = os.path.join(dataset_folder, bug_folder, 'after')
        for code_file in os.listdir(after_folder):
            if code_file.endswith('.py'):
                code_files.append(os.path.join(after_folder, code_file))
    code_files = ast_tokenize(code_files)
    logger.debug("{} code files.".format(len(code_files)))
    train(code_files, vocab_size=vocab_size)


if __name__ == '__main__':
    # dataset_folder = '/run/media/zg/data/Research/AI/BugDetection/smartscript/data'
    # dataset_folder = '/home/zg/Dropbox/phd/projects/SmartScript/cases'
    # args = toml.load('args.toml')
    # dataset_folder = args['dataset']['path']
    # train_on_code_files(dataset_folder)
    test_code = """
    def test():
        # comment
        pass
    """
    tokens = ast_tokenize_str(test_code)
    print(tokens)
