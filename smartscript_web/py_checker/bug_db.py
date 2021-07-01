# encoding=utf-8
"""
In the database we have several collections:
- top_k_code: all the python code of top k python projects in Github
-
"""
import pymongo


def connect(ip='localhost'):
    db = pymongo.MongoClient(ip, 27017, username='<username>',
                             password='<password>',
                             authSource='bug_db',
                             authMechanism='SCRAM-SHA-1')
    return db


if __name__ == '__main__':
    ip = '<IP>'
    db = connect(ip)
    print(db.list_databases())
