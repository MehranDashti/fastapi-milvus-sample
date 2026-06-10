from milvus_client import get_client, ensure_collection, get_collection_stats

client = get_client()
ensure_collection(client)

stats = get_collection_stats(client)
print("\n--- Collection Stats ---")
print("Name      :", stats["collection"])
print("Row count :", stats["row_count"])
print("Fields    :", stats["fields"])