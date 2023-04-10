from heapq import nlargest
import pathlib
from sentence_transformers import SentenceTransformer, util
import datetime
import os
from pathlib import Path
import numpy as np

compare_dict = []


def distrib(compare_dict):
    threshold = 0.2  # порог
    c_num = 3  # количество рецензентов на оценку доклада
    m_dict = {}  # промежут словарь
    final_dict = {}
    for rep in compare_dict:
        max_value = nlargest(c_num, compare_dict[rep].values())  # отбор c_num максимальных знач-й
        m_dict[rep] = {k: v for k, v in compare_dict[rep].items() if v in max_value}  # словарь с макс знач-ми
        final_dict[rep] = {}
        for c_id in m_dict[rep]:  # заполнени словаря, учитывая порог
            if m_dict[rep][c_id] >= threshold:
                final_dict[rep][c_id] = m_dict[rep][c_id]
    return final_dict


def compare(report, article):
    sentences = [report, article]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Compute embedding for both lists
    embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    return float(util.pytorch_cos_sim(embedding_1, embedding_2))


def recur(articlepath, id_report, report):
    arr = np.array([0])
    for root, dirs, files in os.walk(articlepath):
        for directory in dirs:
            for file2 in files:
                with open(root + '\\' + file2, "r") as f:
                    article = f.read()
                # id_article = get_file_id(file2)
                np.append(arr, compare(report, article))
            compare_dict[id_report][directory] = np.max(arr)
            # Similarity.objects.create(id_author=id_report, id_reviewer=directory,
            #           similar=compare_dict[id_report][directory])


def get_file_id(file):  # временно название доклада/статьи - это их id
    return file


def recurtraver(path, articlepath):
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(root + '\\' + file, "r") as f:
                report = f.read()
            id_report = get_file_id(file)
            compare_dict[id_report] = {}
            recur(articlepath, id_report, report)
    final_dict = distrib(compare_dict)
    return final_dict


def distribute():
    # Получаем строку, содержащую путь к рабочей директории:
    dir_path = pathlib.Path.cwd()
    # parent directory
    parent = os.path.join(dir_path, os.pardir)
    today = datetime.datetime.today()
    year = today.strftime("%Y")
    print(os.path.abspath(parent))
    # Объединяем полученную строку с недостающими частями пути
    path = Path(os.path.abspath(parent), 'media', 'report', year)
    articlepath = Path(os.path.abspath(parent), 'media', 'article', year)
    recurtraver(path, articlepath)


distribute()
