from sqlalchemy import Column, String, BigInteger, Text, DateTime, create_engine, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import random
import os
import xlrd

base = declarative_base()


class Author(base):
    __tablename__ = 'scholar_paper_graph'
    paperId = Column('paperId', BigInteger, primary_key=True)
    peopleId = Column('peopleId', BigInteger, primary_key=True)
    scholarNamePinyin = Column('ScholarNamePinyin', String)
    scholarNamePinyin2 = Column('ScholarNamePinyin2', String)
    paperTitle = Column('paperTitle', Text)
    paperTitleLowerCase = Column('paperTitleLowerCase', Text)
    paperVertexId = Column('paperVertexId', BigInteger)
    scholarName = Column('scholarName', String)
    subDepartment = Column('subDepartment', String)
    updateTime = Column('updateTime', DateTime)
    status = Column('status', String)
    isChineseTitle = Column('isChineseTitle', String)


class Paper(base):
    __tablename__ = 'trans_paper'
    id = Column('id', Integer, primary_key=True)
    title = Column('title', String)
    title_en = Column('title_en', String)


# def combination_by_select()


def random_combination(paperIds, paper_infos, author_info):
    result = []
    random_num = 5
    before_ids = []
    negative_ids = []
    random_negative = list(paper_infos.keys())

    for paperId in paperIds:
        if len(before_ids) > 0:
            select_combination_ids = random.sample(before_ids, random_num) if len(
                before_ids) >= random_num else random.sample(before_ids, len(before_ids))
            for _paperId in select_combination_ids:
                # 负样例
                random_id = random.choice(random_negative)
                while not paper_infos.__contains__(random_id) or random_id in paperIds:
                    random_id = random.choice(random_negative)
                if paper_infos.__contains__(_paperId) and paper_infos.__contains__(paperId):
                    # 中文-中文
                    result.append({
                        'title': paper_infos[paperId].get('title'),
                        'title_en': paper_infos[_paperId].get('title'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 1
                    })
                    # 英文-英文
                    result.append({
                        'title': paper_infos[paperId].get('titleEn'),
                        'title_en': paper_infos[_paperId].get('titleEn'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 1
                    })
                    # 中文-英文
                    result.append({
                        'title': paper_infos[paperId].get('title'),
                        'title_en': paper_infos[_paperId].get('titleEn'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 1
                    })
                    # 英文-中文
                    result.append({
                        'title': paper_infos[paperId].get('titleEn'),
                        'title_en': paper_infos[_paperId].get('title'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 1
                    })
                    # ==============================负样例=============================
                    # 中文-中文
                    result.append({
                        'title': paper_infos[paperId].get('title'),
                        'title_en': paper_infos[random_id].get('title'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 0
                    })
                    # 英文-英文
                    result.append({
                        'title': paper_infos[paperId].get('titleEn'),
                        'title_en': paper_infos[random_id].get('titleEn'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 0
                    })
                    # 中文-英文
                    result.append({
                        'title': paper_infos[paperId].get('title'),
                        'title_en': paper_infos[random_id].get('titleEn'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 0
                    })
                    # 英文-中文
                    result.append({
                        'title': paper_infos[paperId].get('titleEn'),
                        'title_en': paper_infos[random_id].get('title'),
                        'scholarName': author_info['scholarName'],
                        'scholarNamePinyin': author_info['scholarNamePinyin'],
                        'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                        'label': 0
                    })
        if paper_infos.__contains__(paperId):
            # 正样例
            result.append({
                'title': paper_infos[paperId].get('title'),
                'title_en': paper_infos[paperId].get('titleEn'),
                'scholarName': author_info['scholarName'],
                'scholarNamePinyin': author_info['scholarNamePinyin'],
                'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                'label': 1
            })
            # 负样例
            random_id = random.choice(random_negative)
            while not paper_infos.__contains__(random_id):
                random_id = random.choice(random_negative)
            result.append({
                'title': paper_infos[paperId].get('title'),
                'title_en': paper_infos[random_id].get('titleEn'),
                'scholarName': author_info['scholarName'],
                'scholarNamePinyin': author_info['scholarNamePinyin'],
                'scholarNamePinyin2': author_info['scholarNamePinyin2'],
                'label': 0
            })
            before_ids.append(paperId)
    return result


def main():
    # workboot = xlrd.open_workbook(r'F:\Python\Git\My\NLPProject\Kejso\org_aliases_new(1).xlsx')
    # booksheet = workboot.sheet_by_name('org_aliases_new')
    # p = list()
    # for row in range(1, booksheet.nrows):
    #     row_data = []
    #     # for col in range(booksheet.ncols):
    #
    #     for col in [1, 2, 4, 5, 6]:
    #         val = booksheet.cell(row, col).value
    #     val = cel.value
    #     print(val)
    #     cel = booksheet.cell(row, 2)
    #     val = cel.value
    #     print(val)
    #     cel = booksheet.cell(row, 4)
    #     val = cel.value
    #     print(val)
    #     cel = booksheet.cell(row, 5)
    #     val = cel.value
    #     print(val)
    #     cel = booksheet.cell(row, 6)
    #     val = cel.value
    #     print(val)

    engine = create_engine('mysql+mysqlconnector://haochengqian:Ww800880!@123.57.58.2:3306/haochengqian?charset=utf8')
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    author_list = session.query(Author).filter(Author.scholarName != None, Author.scholarNamePinyin2 != None,
                                               Author.scholarNamePinyin != None).all()
    author_paperids = {}
    for author in author_list:
        if author_paperids.__contains__(author.peopleId):
            c = author_paperids[author.peopleId]
            d = c['paperIds']
            d.append(author.paperId)
            # author_paperids[author.paperId]['paperIds'].append(author.paperId)
        else:
            author_paperids[author.peopleId] = {
                'scholarName': author.scholarName,
                'scholarNamePinyin': author.scholarNamePinyin,
                'scholarNamePinyin2': author.scholarNamePinyin2,
                'paperIds': [author.paperId]
            }
    paper_list = session.query(Paper).filter(Paper.title is not None, Paper.title_en is not None).all()
    paper_infos = {}
    for paper in paper_list:
        paper_infos[paper.id] = {
            'title': paper.title,
            'titleEn': paper.title_en,
        }

    save_file_infos = []
    save_path = './data/'
    save_all_file_path = './data/allData/all_data.txt'
    for key, info in author_paperids.items():
        if len(info['paperIds']) > 2:
            c = random_combination(info['paperIds'], paper_infos, info)
            save_file_infos.extend(c)
        else:
            _paper_id = info['paperIds'][0]
            if paper_infos.__contains__(info['paperIds'][0]):
                save_file_infos.append({

                    'title': paper_infos[info['paperIds'][0]].get('title'),
                    'title_en': paper_infos[_paper_id].get('title_en'),
                    'scholarName': info['scholarName'],
                    'scholarNamePinyin': info['scholarNamePinyin'],
                    'scholarNamePinyin2': info['scholarNamePinyin2'],
                    'label': 1
                })
    with open(save_all_file_path, 'w', encoding='utf-8') as f:
        random.shuffle(save_file_infos)
        for index, p in enumerate(save_file_infos):
            f.write(
                f"{p['title']}\t\1\t{p['title_en']}\t\1\t{p['scholarName']}\t\1\t{p['scholarNamePinyin']}\t\1\t{p['scholarNamePinyin2']}\t\1\t{p['label']}\n")

    train_per = 0.8
    valid_per = 0.1
    random.shuffle(save_file_infos)

    all_data_len = len(save_file_infos)

    train_f = open(os.path.join(save_path, 'train.txt'), "w", encoding='utf-8')
    valid_f = open(os.path.join(save_path, 'valid.txt'), "w", encoding='utf-8')
    test_f = open(os.path.join(save_path, 'Test.txt'), "w", encoding='utf-8')
    for index, p in enumerate(save_file_infos):
        if index <= all_data_len * train_per:
            train_f.write(
                f"{p['title']}\t\1\t{p['title_en']}\t\1\t{p['scholarName']}\t\1\t{p['scholarNamePinyin']}\t\1\t{p['scholarNamePinyin2']}\t\1\t{p['label']}\n")
        elif index < all_data_len * (train_per + valid_per):
            valid_f.write(
                f"{p['title']}\t\1\t{p['title_en']}\t\1\t{p['scholarName']}\t\1\t{p['scholarNamePinyin']}\t\1\t{p['scholarNamePinyin2']}\t\1\t{p['label']}\n")
        else:
            test_f.write(
                f"{p['title']}\t\1\t{p['title_en']}\t\1\t{p['scholarName']}\t\1\t{p['scholarNamePinyin']}\t\1\t{p['scholarNamePinyin2']}\t\1\t{p['label']}\n")


if __name__ == '__main__':
    main()
