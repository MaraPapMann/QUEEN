"""
@Author: M.H.C.
@Desc: A .json parser
"""
import json


def parse_json(fp:str) -> dict:
    with open(fp, 'r') as f:
        d = json.load(f)
    d = json.loads(d)
    return d


def write_json(fp:str, d:dict) -> None:
    str_d = json.dumps(d)
    with open(fp, 'w') as f:
        json.dump(str_d, f)
    f.close()
    return


# Debug
if __name__ == '__main__':
    d = parse_json('configs/cifar_wide_resnet.json')
    print(d)