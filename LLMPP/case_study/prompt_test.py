import requests
import json
import time

prompt_initial = """
You are an expert in mobile app privacy policy analyzer and have over ten years experience in analyzing the compliance in privacy policies.

You will be given one sentence from a Chinese privacy policy (PP). You should analyze the  sentence to identify the key components in the privacy policy. Specifically, you need to identify the data controller, data entities, data behavior (collect or share), condition, purpose and data receiver.  The definition of the seven components are 
as follows: 

1. Data controller: (noun) the party that determines the purpose and means of personal data processing. 
2. Data entities: (nouns) any information that can identify or reflect the activities of a natural person.
3. Collection: verbs that describe how controllers access data.
4. Sharing: verbs that describe whether how the controller distribute the entities to others.
5. Condition: the situation where the data controller will access personal data. 
6. Purpose: why the data controller processes user data. 
7. Data receiver: the parties that receive user data. 
0. Others: that are not related to aforementioned components.

Your need to output the label of each character in the provided sentence. The length of output label must be strictly the same with the character length of provide sentence. For example:
Example 1:
"sentence": "您在注册启信宝时，我们会收集、使用、保存、共享您的相关个人信息.".
"label": "0 5 5 5 5 5 5 0 0 1 1 0 3 3 0 3 3 0 3 3 0 4 4 0 0 0 0 2 2 2 2 0"
Example 1:
"sentence": "我们可能会为了推荐适合您的产品与供应商共享您的兴趣爱好。"
"label": "1 1 0 0 6 6 6 6 6 6 6 6 6 6 0 7 7 7 5 5 0 0 2 2 2 2 0""


Now, it is your turn:
"sentence": "{sentence}"
"label": """""

sentences = [
    "四、用户个人资料\n**用户同意：**\n**提供及时、详尽及准确的个人资料；在百果园提出要求时，按照符合法律规定和行业标准的方式进行实名认证或以其他方式提供全面、真实、有效的身份信息。",
    "您还可以根据自身需求选择填写性别、生日、地区及个人介绍来完善您的信息。",
    "在您使用身份认证的功能或相关服务所需时，根据相关法律法规，您可能需要提供您的真实身份信息（真实姓名、身份证号码、电话号码）以完成实名验证。\n该等信息属于个人敏感信息，您可以拒绝提供，如果拒绝提供您将可能无法获得相关服务，但不影响其他功能与服务的正常使用。",
    "未经您许可，梦映不会向第三方公开、透露您的个人信息。",
    "7、    为了改善武汉氪细胞的技术和服务，向用户提供更好的服务体验，武汉氪细胞或可会自行收集使用或向第三方提供用户的非个人隐私信息。"
]


def query_gpt35(prompt):
    url = "https://oa.api2d.net/v1/chat/completions"

    payload = json.dumps({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "safe_mode": False
    })
    headers = {
        'Authorization': 'Bearer fk229918-IDyuH0Far2THEUMDUcOkluQMgxqDEecl',
        'Content-Type': 'application/json'
    }

    beg_time = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    duration = time.time() - beg_time
    response_content = json.loads(response.text)
    if "choices" not in response_content:
        return "", duration
    response_content = response_content['choices'][0]['message']['content']
    # print(f"Duration:{duration} ::: {response_content}")
    return response_content, duration


for sentence in sentences:
    prompt = prompt_initial.format(sentence=sentence)
    # print(prompt)
    resp, duration = query_gpt35(prompt)
    print(f"{'=' * 40}")
    print(f"sentence: {sentence}\n resp: {resp}")
    print(f"{'=' * 40}")
