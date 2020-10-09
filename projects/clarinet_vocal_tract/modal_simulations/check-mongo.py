from sshtunnel import SSHTunnelForwarder
import pymongo
import os
import stat
import json
import datetime
import paramiko
import pandas

pkey=os.path.join(os.environ['HOME'],'.ssh/id_rsa')
key=paramiko.RSAKey.from_private_key_file(pkey)

MONGO_HOST = "129.94.162.112"
MONGO_USER = "goios"
MONGO_DB = "test2"
MONGO_COLLECTION = "ls-remote"
local_port = 26017

MONGO_DB = "modal-2duct-simulations"
#MONGO_DB = "test2"
MONGO_COLLECTION = "random-runs-var-gamma-pert-time"

with SSHTunnelForwarder(    
    MONGO_HOST,
    ssh_username=MONGO_USER,
    ssh_pkey="~/.ssh/id_rsa",
    remote_bind_address=('localhost', 27017),
    local_bind_address=('localhost',local_port)
) as server:
    with pymongo.MongoClient('localhost', local_port) as connection:
        db = connection[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        print (db)
        print("List of DBs on mongo:")
        print(json.dumps(connection.list_database_names(),indent=2))
        print("List of collections in db %s:"%(MONGO_DB))
        print(json.dumps(db.list_collection_names(), indent=2))    

        db = connection[MONGO_DB]

        for collection_name in db.list_collection_names():
        
            collection = db[collection_name]
            print("Number of documents in %s: %d"%(collection_name,collection.count()))
            print("First date: %s"%(collection.find_one(sort=[("simulation.start",1)])["simulation"]["start"]))
            print("Last date: %s"%(collection.find_one(sort=[("simulation.start",-1)])["simulation"]["start"]))