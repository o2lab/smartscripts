import time
import pandas
import pymongo
from GenerateCurve import updateWeb
class CollectionsStat:

    MONGODB_URI = "mongodb://127.0.0.1:27017"
    DATABASE__STAT_INDEX_ALL = ["raw", "objects", "avgObjSize", "dataSize", "storageSize", "numExtents", "indexes",
                                "indexSize", "fileSize", "extentFreeList"]
    DATABASE__STAT_INDEX = ["objects", "avgObjSize", "dataSize", "storageSize", "numExtents", "indexes", "indexSize",
                            "fileSize"]
    COLLECTION__STAT_INDEX_ALL = ["ns", "sharded", "capped", "count", "size", "storageSize", "totalIndexSize",
                                  "indexSizes",
                                  "avgObjSize", "nindexes", "nchunks", "shards"]
    COLLECTION__STAT_INDEX = ["ns", "sharded", "capped", "count", "size", "storageSize", "totalIndexSize", "avgObjSize",
                              "nindexes", "nchunks"]

    def __init__(self, db_name):
        self.client = db = pymongo.MongoClient("127.0.0.1", 27017, username='<username>',
                             password='<password>',
                             authSource='bug_db',
                             authMechanism='SCRAM-SHA-1')
        self.database = self.client.get_database(db_name)
#         self.database.au
#         print("Connected")
    def connect(self, db_name):
        self.client = db = pymongo.MongoClient("127.0.0.1", 27017, username='<username>',
                             password='<password>',
                             authSource='bug_db',
                             authMechanism='SCRAM-SHA-1')
        self.database = self.client.get_database(db_name)
#         self.database.au
#         print("Connected")

    def get_db_stat(self):
        # 输出数据库统计
        db_cursor = self.database.command("dbstats")  # type:dict
        db_data = {}
        for ele in self.DATABASE__STAT_INDEX:
            db_data[ele] = db_cursor[ele]
        print(db_data)

    def get_coll_stat(self):
        # 集合统计
        coll_cursor_list = self.database.command("listCollections")["cursor"]["firstBatch"]
        coll_data = {}
        for ele in self.COLLECTION__STAT_INDEX:
            coll_data[ele] = []
        for coll_ele in coll_cursor_list:
            collections_name = coll_ele["name"]
            coll_stat = self.database.command("collstats", collections_name)  # type:dict
            for ele in self.COLLECTION__STAT_INDEX:
                if ele in coll_stat.keys():
                    coll_data[ele].append(coll_stat[ele])
                else:
                    coll_data[ele].append(0)
        # 将集合统计结果转为DataFrame
        self.coll_df = pandas.DataFrame(coll_data)
        # 获取当前时间
        current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        result_path = "./coll_stat/coll_stat_%s.csv" % current_time
        # 输出到文件
        self.coll_df.to_csv(result_path, index=False)
        
    def update_coll_stat(self):
        # 集合统计
        coll_cursor_list = self.database.command("listCollections")["cursor"]["firstBatch"]
        coll_data = {}
        for ele in self.COLLECTION__STAT_INDEX:
            coll_data[ele] = []
        for coll_ele in coll_cursor_list:
            collections_name = coll_ele["name"]
            coll_stat = self.database.command("collstats", collections_name)  # type:dict
            for ele in self.COLLECTION__STAT_INDEX:
                if ele in coll_stat.keys():
                    coll_data[ele].append(coll_stat[ele])
                else:
                    coll_data[ele].append(0)
        # 将集合统计结果转为DataFrame
        coll_df0 = pandas.DataFrame(coll_data)
#         print()
        return coll_df0.equals(self.coll_df)
#         print((coll_df0==self.coll_df).all())
#         # 获取当前时间
#         current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
#         result_path = "./coll_stat/coll_stat_%s.csv" % current_time
#         # 输出到文件
#         coll_df.to_csv(result_path, index=False)

    def __del__(self):
#         print("Disconnect")
        self.client.close()


if __name__ == "__main__":
    collection_stat = CollectionsStat("bug_db")
    collection_stat.get_coll_stat()
    collection_stat.__del__()
    updateWeb()
    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    print("Program starts at ",current_time)
    while(True):
        collection_stat.connect("bug_db")
#         collection_stat.get_coll_stat()
        if not (collection_stat.update_coll_stat()):
            current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            updateWeb()
            print("Database updated at ",current_time)
            collection_stat.get_coll_stat()
        collection_stat.__del__()
        time.sleep(86400)
   # collection_stat.get_coll_stat()