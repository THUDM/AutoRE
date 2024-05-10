"""
Description: 
Author: dante
Created on: 2023/10/18
"""
import random
import time
import requests
from basic import *


keys_file = "keys.json"
current_dir = os.path.dirname(os.path.abspath(__file__))
ori_keys = json.load(open(os.path.join(current_dir, f"../data/chatgpt_count/{keys_file}")))
keys = [key for key, v in ori_keys.items() if v['label']]
unused_keys = keys.copy()
used_keys = []
overload_keys = []
overload_keys_200 = []
invalid_keys = []
# for proxies, you can comment it out if you do not need it, and you should change it to your env setting
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


def gen_analysis(sample, save_file):
    # to make analysis dataset for redocred
    sample['relation_analysis'] = {}
    sample["entity_analysis"] = {}
    relations_desc = [relations_description.get(relation) for relation in sample['relations']]
    prompt = f"Given the passage: {sample['passage']}, after analyzing the text, we have identified the relations: {sample['relations']}, the specific relation descriptions are as " \
             f"follows: {relations_desc}.\n Now, provide me with the analysis. Your analysis should be short but convincing. You can start with : according to the passage, " \
             f"the relations are ... the reasons are...\n" \
             f"You should analyze every relation.\n" \
             f"You should focus on the evidences that led to the conclusion."
    analysis = make_chat_request(prompt)
    sample['relation_analysis'] = analysis.replace("\n\n", "\n")
    sample["fact_analysis"] = {}
    for relation in sample['relations']:
        entity_list = list(set([fact[0] for fact in sample['fact_list'] if fact[1] == relation]))
        if not entity_list:
            continue
        entity_prompt = f"You are an expert in entity analysis, you have been presented with a passage:\"{sample['passage']}\". From this passage, we can derive the " \
                        f"relation: \"{relation}\". The description of this relation is: \"{relations_description.get(relation)}\". Based on the information in the " \
                        f"passage and the relation description, we have identified the following entities as the subjects of the fact related to \"{relation}\": " \
                        f"{entity_list}. Now, please explain why these entities can be considered as the subjects of the fact related to \"{relation}\". Your explanations " \
                        f"should be succinct yet persuasive."
        analysis = make_chat_request(entity_prompt)
        sample["entity_analysis"][relation] = analysis.replace("\n\n", "\n")
        sample["fact_analysis"][relation] = {}
        fact_analysis = {}
        for entity in entity_list:
            fact_list = [fact for fact in sample['fact_list'] if fact[1] == relation and fact[0] == entity]
            fact_prompt = f"You are a fact analysis expert.\n" \
                          f"I have passage : \"{sample['passage']}\"\n" \
                          f"The relation description is: \"{relations_description.get(relation)}\"\n" \
                          f"To extract facts of \"{relation}\", we make \"{entity}\" as subject according to the relation description, " \
                          f"after carefully analysing the passage, we get the fact: {fact_list}. " \
                          f"Now give me the analysis. Your analysis should be short but convincing. You can start with:\n" \
                          f"according to the {entity} and {relation} , the facts are..., the reasons are.."

            analysis = make_chat_request(fact_prompt)
            fact_analysis[entity] = analysis.replace("\n\n", "\n")
        sample["fact_analysis"][relation] = fact_analysis
    with open(save_file, "a") as f:
        print(f"{sample['index']} write to {save_file}")
        f.write(json.dumps(sample) + "\n")
        update_keys_file()

