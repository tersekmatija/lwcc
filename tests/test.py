from src.lwcc import LWCC

# Image paths
img1 = "dataset/img01.jpg"
img2 = "dataset/img02.jpg"

# Initialize model and predict
#model = LWCC.load_model(model_weights= "SHA")
#count = LWCC.get_count(img2, model = model)
#print(f"Count for img1: {count}")

# Direct
#count = LWCC.get_count(img2)
#print(f"Count for img1: {count}")

# Test multiple
counts = LWCC.get_count([img1, img2], model_name= "SFANet", model_weights="SHB")
print(f"Counts for img1, img2: {counts}")