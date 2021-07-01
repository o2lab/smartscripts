""" format string bug """
import sys

_ = lambda s: s


def log(fout, msg, **kwargs):
    """ Simple log function """
    fout.write(msg.format(**kwargs))


if __name__ == '__main__':
    log(sys.stdout, _("Is it a bug? {result}\n"), result='yes')
    print(_(
        '%(token)s: has %(bug)s'
    ) % {
        'token': 'test',
        'bug': 'no'
    })
