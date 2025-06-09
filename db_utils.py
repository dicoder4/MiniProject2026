# db_utils.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()  

def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    return MongoClient(mongo_uri)

def get_users_collection():
    client = get_mongo_client()
    db = client["flood_evacuation"] 
    return db["users"]   
 
def save_user(user):
    users_col = get_users_collection()
    users_col.update_one({"username": user["username"]}, {"$set": user}, upsert=True)   
    
def get_all_users():
    users_col = get_users_collection()
    return list(users_col.find({}))        