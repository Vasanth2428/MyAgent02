import weaviate.classes as wvc

print("VectorIndex methods:")
for x in dir(wvc.config.Configure.VectorIndex):
    if not x.startswith("_"):
        print(f"  {x}")

print("\nVectorIndex config classes:")
for x in dir(wvc.config):
    if "VectorIndex" in x:
        print(f"  {x}")
