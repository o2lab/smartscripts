""" format string bug
https://github.com/luispedro/django-gitcms/commit/0a6449d5641cc510d11ae314fb68e4f9848261f2
https://github.com/rheimbuch/Arelle/commit/1cb73c5aa33b14f69564e3a144302ee5a668d84e """
import sys

_ = lambda s: s


def log(fout, msg, **kwargs):
    """ Simple log function """
    fout.write(msg.format(**kwargs))


if __name__ == '__main__':
    log(sys.stdout, _("Is it a bug? {result}\n"))
    print(_(
        '%(token)s: has %(bug}s'
    ) % {
        'token': 'test',
        'bug': 'no'
    })
