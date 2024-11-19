from typing import Dict


def type_judge(final: Dict):
    result = {}
    catagory = [*final]
    exclude_values = [value for k, value in final.items() if k != "平和体质"]
    if '平和体质' in catagory:
        score_ph = final["平和体质"]
        if score_ph > 60 and all(x < 30 for x in exclude_values):
            result["平和体质"] = 2
        elif score_ph > 60 and all(x < 40 for x in exclude_values):
            result["平和体质"] = 1
        else:
            result["平和体质"] = 0
    else:
        raise ValueError('Missing key "平和体质"')
    
    catagory.remove("平和体质")
    for cat in catagory:
        score = final[cat]
        if score >= 40:
            result[cat] = 2
        elif score < 40 and score >= 30:
            result[cat] = 1
        elif score < 30:
            result[cat] = 0
        else:
            raise ValueError(f'Wrong value of type {cat}')
    return result
