import requests

# API URL
url = "http://127.0.0.1:5000/recognize"

# Correct path to the image (use one of the fixes above)
file_path = r"C:\Users\divya chaudhary\Desktop\python\Face-recognition-bug-fixes\Images\divya.jpg"

# Open the image and send it as a POST request
with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})
    

# Print the API response
print(response.json())
