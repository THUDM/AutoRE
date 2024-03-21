"""
Description: 
Author: dante
Created on: 2023/11/15
"""

templates = {
    "D_F": {
        "fact_list_template": "Given a passage: \"{sentences}\"\nList all underlying facts."
    },
    "D_RS_F": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList all underlying relations.",
        "fact_list_template": "Given a passage: \"{sentences}\"\nList all triplet facts according to the relation list : \"{relations}\"n."
    },
    "D_R_F": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList all underlying relations.",
        "fact_list_template": "Given a passage: \"{sentences}\"\nList all triplet facts that take \"{relation}\" as the relation."
    },
    "D_R_H_F": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList all underlying relations.",
        "entity_list_template": "Given a relation : \"{relation}\" and a passage: \"{sentences}\"\nlist the entities that can be serve as suitable subjects for the relation.",
        "fact_list_template": "Provided a passage: \"{sentences}\"\nList all triplet facts that take \"{relation}\" as the relation and \"{subject}\" as the subject."
    },
    "D_R_H_F_desc": {
        "relation_list_template": "Given a passage: \"{sentences}\"\nList any underlying relations.",
        "entity_list_template": "Given a relation \"{relation}\", and its description: \"{description}\" and a passage: \"{sentences}\", list entities that can be identified as "
                                "suitable subjects for the relation.",
        "fact_list_template": "Given relation \"{relation}\" and relation description: \"{description}\".\n"
                              "Provided a passage: \"{sentences}\"\n"
                              "List all triplet facts that take \"{relation}\" as the relation and \"{subject}\" as the subject."
    }
}
