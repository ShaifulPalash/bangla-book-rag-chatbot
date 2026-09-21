from src.embeddings import get_embedding_model

print("First call")
model1 = get_embedding_model()

print("Second call")
model2 = get_embedding_model()

print("Same object:", model1 is model2)