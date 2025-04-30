import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.json_util import dumps  # handles ObjectId, dates, etc.

def export_single_document(doc_id: str, output_path: str):
    load_dotenv()
    uri        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name    = os.getenv("MONGODB_DB", "clm_db")
    col_name   = os.getenv("MONGODB_COLLECTION", "documents")

    client = MongoClient(uri)
    db     = client[db_name]
    col    = db[col_name]

    # Fetch one document
    doc = col.find_one({"id": doc_id})
    if not doc:
        print(f"[!] No document found with id={doc_id}")
        return

    # Write it out as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc, indent=2))

    print(f"[✓] Exported document {doc_id} → {output_path}")
    client.close()


def export_entire_collection(output_path: str):
    load_dotenv()
    uri        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name    = os.getenv("MONGODB_DB", "clm_db")
    col_name   = os.getenv("MONGODB_COLLECTION", "documents")

    client = MongoClient(uri)
    db     = client[db_name]
    col    = db[col_name]

    # Fetch all documents
    cursor = col.find()
    docs = list(cursor)  # pull into a list

    # Write list out as a JSON array
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dumps(docs, indent=2))

    print(f"[✓] Exported {len(docs)} documents → {output_path}")
    client.close()


if __name__ == "__main__":
    # 1) To export just one document by ID:
    export_single_document(
        doc_id="64f1e3a2b9da4c0012345678",
        output_path="single_contract.json"
    )

    # 2) To export the entire collection:
    export_entire_collection(
        output_path="all_contracts.json"
    )
