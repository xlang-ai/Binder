import requests
import base64
import time


def vqa_call(question, image_path, api_url='https://hf.space/embed/OFA-Sys/OFA-vqa/+/api/predict/'):
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read())
    base64_data_to_send = "data:image/{};base64,{}".format(image_path.split(".")[-1], str(base64_data)[2:-1])
    return requests.post(url=api_url, json={"data": [base64_data_to_send, question]}).json()['data'][0]

if __name__ == '__main__':
    # demo code
    image_path = "0af77d205dc6673001cdd9ea753f880e.JPG"
    question = "what is the dressing of that woman?"
    api_url = 'https://hf.space/embed/OFA-Sys/OFA-vqa/+/api/predict/'
    result = vqa_call(question, image_path, api_url)
    print(result)
