import os

print("1 Train")
print("2 Predict")
print("3 Explain")

choice = input("Enter: ")

if choice == "1":
    os.system("python src/train.py")
elif choice == "2":
    os.system("python src/predict.py")
elif choice == "3":
    os.system("python src/explain.py")