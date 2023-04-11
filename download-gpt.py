import urllib.request

# download gpt-merges.txt
url1 = 'https://huggingface.co/gpt2/raw/main/merges.txt'
filename1 = 'gpt-merges.txt'
urllib.request.urlretrieve(url1, filename1)

# download gpt-vocab.json
url2 = 'https://huggingface.co/gpt2/raw/main/vocab.json'
filename2 = 'gpt-vocab.json'
urllib.request.urlretrieve(url2, filename2)

# download gpt2-small.tflite
url3 = 'https://huggingface.co/gpt2/resolve/main/64-8bits.tflite'
filename3 = 'gpt2-small.tflite'
urllib.request.urlretrieve(url3, filename3)
