from sentence_transformers import SentenceTransformer, util
import datetime
import os
from heapq import nlargest
import pathlib
from pathlib import Path
from pprint import pprint

compare_dict = {}


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


def get_file_id(name):  #временно название доклада/статьи - это их id
    return name


def compare(report, article):
    sentences = [report, article]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Compute embedding for both lists
    embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model.encode(sentences[1], convert_to_tensor=True)

    return float(util.pytorch_cos_sim(embedding_1, embedding_2))


def recurtraver(path, articlepath):
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(root + '\\' + file, "r") as f:
                report = f.read()
            id_report = get_file_id(file)
            compare_dict[id_report] = {}
            recur(articlepath, id_report, report)
    pprint(distrib(compare_dict))


def recur(articlepath, id_report, report):
    for root, dirs, files in os.walk(articlepath):
        for file2 in files:
            with open(root + '\\' + file2, "r") as f:
                article = f.read()
            id_article = get_file_id(file2)
            compare_dict[id_report][id_article] = compare(report, article)


def distribute():
    # Получаем строку, содержащую путь к рабочей директории:
    dir_path = pathlib.Path.cwd()
    today = datetime.datetime.today()
    year = today.strftime("%Y")
    # Объединяем полученную строку с недостающими частями пути
    path = Path(dir_path, 'media', 'report', year)
    articlepath = Path(dir_path, 'media', 'article', year)
    recurtraver(path, articlepath)


distribute()


"""
def to_send(final_dict):  #
    c_w_set = {}
    for rep in final_dict:
        for c_id in final_dict[rep]:
            c_w_set.add(c_id)
    c_set = {}
    for c in c_w_set:
        c_set.add(get_reviewer_id_by_art_id(c))
    send_dict = {}
    for c in c_set:
        send_dict[c] = []
        for rep in final_dict:
            for c_id in final_dict[rep]:
                if c == get_reviewer_id_by_art_id(c_id):
                    send_dict[c].append(rep)  # вместо rep можно название работы и т.д.
                    break
    return send_dict


def get_reviewer_id_by_art_id(id_article):  # по id статьи возвращает id рецензента
    return 0
    #return id_reviewer
"""
