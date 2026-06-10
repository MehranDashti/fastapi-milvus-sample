from ingester import ingest_file
from milvus_client import get_client, get_collection_stats

# Ingest the sample file
result = ingest_file("sample.txt")

print("\n--- Ingest Result ---")
print("Source  :", result["source"])
print("Chunks  :", result["chunks"])
print("Inserted:", result["inserted"])

# Force flush before checking stats
client = get_client()
client.flush(collection_name="documents")   # ← wait for disk write

stats = get_collection_stats(client)
print("\n--- Collection Stats After Ingest ---")
print("Row count:", stats["row_count"])