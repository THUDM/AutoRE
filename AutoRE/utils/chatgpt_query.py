"""
Description: 
Author: dante
Created on: 2023/10/18
"""
import json
import random
import time
import os
import requests

keys_file = "keys.json"
current_dir = os.path.dirname(os.path.abspath(__file__))
ori_keys = json.load(open(os.path.join(current_dir, f"../data/chatgpt_count/{keys_file}")))
keys = [key for key, v in ori_keys.items() if v['label']]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
overload_keys_200 = []
invalid_keys = []
proxies = {
    'http': '127.0.0.1:9898',
    'https': '127.0.0.1:9898',
}

def get_valid_key():
    global unused_keys, used_keys, overload_keys, overload_keys_200
    current_time = time.time()
    new_overload_keys = []
    for key, timestamp in overload_keys:
        if current_time - timestamp >= 60:
            unused_keys.append(key)
        else:
            new_overload_keys.append((key, timestamp))
    overload_keys = new_overload_keys

    new_overload_keys = []
    for key, timestamp in overload_keys_200:
        if current_time - timestamp >= 70 * 7:
            unused_keys.append(key)
        else:
            new_overload_keys.append((key, timestamp))
    overload_keys_200 = new_overload_keys

    while not unused_keys:
        time.sleep(20)
    key = random.choice(unused_keys)
    unused_keys.remove(key)
    used_keys.append(key)
    return key


def update_keys_file():
    global invalid_keys
    if invalid_keys:
        for invalid_key in invalid_keys:
            ori_keys[invalid_key]['label'] = False
        invalid_keys = []
        json.dump(ori_keys, open(f"/workspace/xll/analysis_kg/public_data/chatgpt_count/{keys_file}", "w"), indent=4)


def make_chat_request(prompt):
    message = [
        {"role": "user", "content": prompt}
    ]
    global unused_keys, used_keys, overload_keys, overload_keys_200
    while True:
        key = get_valid_key()
        try:
            resp = requests.post(
                url=f"https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "temperature": 1.0,
                    "messages": message,
                },
                proxies=proxies
            )
            if resp.status_code == 200:
                used_keys.remove(key)
                unused_keys.append(key)
                # print("ok", json.loads(resp.content)['choices'][0]['message']['content'])
                return json.loads(resp.content)['choices'][0]['message']['content']
            else:
                # print("not ok", key, json.loads(resp.content))
                try:
                    if json.loads(resp.content).get('error'):
                        if json.loads(resp.content).get('error')['message'] == "You exceeded your current quota, please check your plan and billing details.":
                            used_keys.remove(key)
                            invalid_keys.append(key)
                        elif "Your account is not active" in json.loads(resp.content).get('error')['message']:
                            used_keys.remove(key)
                            invalid_keys.append(key)
                        elif "Limit: 200 / day. Please try again in 7m12s." in json.loads(resp.content).get('error')['message']:
                            used_keys.remove(key)
                            overload_keys_200.append((key, time.time()))
                        elif "Limit 200, Used 200, Requested 1. Please try again in 7m12s." in json.loads(resp.content).get('error')['message']:
                            used_keys.remove(key)
                            overload_keys_200.append((key, time.time()))
                        elif "The OpenAI account associated with this API key has been deactivated" in json.loads(resp.content).get('error')['message']:
                            used_keys.remove(key)
                            invalid_keys.append(key)
                        else:
                            used_keys.remove(key)
                            overload_keys.append((key, time.time()))
                    else:
                        print("response error: ", resp.content)
                        used_keys.remove(key)
                        overload_keys.append((key, time.time()))
                except:
                    print("error: ", key, resp.content)
        except Exception as e:
            print(e)
            used_keys.remove(key)
            overload_keys.append((key, time.time()))
