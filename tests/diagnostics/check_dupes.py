import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from collections import Counter

load_dotenv()

url = os.getenv("WEAVIATE_URL")
api_key = os.getenv("WEAVIATE_API_KEY")

def check_duplicates():
    print(f"Connecting to {url}...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key),
        )
        
        collection = client.collections.get("RAGKnowledge")
        
        # Fetch all objects
        print("Fetching all objects from RAGKnowledge...")
        all_objects = []
        for obj in collection.iterator():
            all_objects.append(obj.properties.get("text", ""))
        
        total_count = len(all_objects)
        print(f"Total objects fetched: {total_count}")
        
        # Exact match deduplication
        counts = Counter(all_objects)
        duplicates = {text: count for text, count in counts.items() if count > 1}
        
        unique_count = len(counts)
        duplicate_entries_count = sum(count for count in duplicates.values())
        
        print("\n--- Results ---")
        print(f"Total Objects: {total_count}")
        print(f"Unique Objects: {unique_count}")
        print(f"Duplicate Objects (Total count of non-unique entries): {duplicate_entries_count}")
        print(f"Redundant objects: {total_count - unique_count}")
        
        if duplicates:
            print("\nTop Duplicates (Sample):")
            for i, (text, count) in enumerate(sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]):
                snippet = text[:100].replace("\n", " ")
                print(f"{i+1}. Found {count} times: \"{snippet}...\"")
        else:
            print("No exact duplicates found.")
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_duplicates()
