import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth

load_dotenv()

url = os.getenv("WEAVIATE_URL")
api_key = os.getenv("WEAVIATE_API_KEY")

print(f"Connecting to {url}...")
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
    )
    
    collection = client.collections.get("RAGKnowledge")
    count = collection.aggregate.over_all(total_count=True).total_count
    print(f"Total objects in RAGKnowledge: {count}")
    
    # Check if we can retrieve anything
    response = collection.query.fetch_objects(limit=5)
    print(f"Sample objects: {len(response.objects)}")
    for obj in response.objects:
        print(f"- {obj.properties.get('text')[:50]}...")
    
    client.close()
except Exception as e:
    print(f"Error: {e}")
