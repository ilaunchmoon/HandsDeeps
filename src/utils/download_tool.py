import hashlib
import os 
import tarfile
import zipfile
import requests
import pandas as pd


DATA_HUB = dict()
DATA_URL = 'https://d2l-data.s3-accelerate.amazonaws.com/'
def download(name, cache_dir=os.path.join('.', 'data')):
    assert name in DATA_HUB, f"{name} not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'downloading {fname} from the {url}')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'only zip/tar file can uncompressing'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    for name in DATA_HUB:
        download(name)


if __name__ == "__main__":
    DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', 
                                      '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 
                                     'fa19780a7b011d9009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(download('kaggle_house_train'))
    test_data = pd.read_csv(download('kaggle_house_test'))  
    print(train_data.shape)
    print(test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
    

