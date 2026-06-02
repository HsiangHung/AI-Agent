
from datetime import datetime
import requests


# 第一个工具，查票
def check_tick(date, start, end):
    print('开始访问12306接口:',date, start, end)
    url = 'https://kyfw.12306.cn/otn/leftTicket/queryG?leftTicketDTO.train_date={}&leftTicketDTO.from_station={}&leftTicketDTO.to_station={}&purpose_codes=ADULT'.format(
        date, start, end)

    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "If-Modified-Since": "0",
        "Referer": "^B8^%^8A^%^E6^%^B5^%^B7,SHH&ts=^%^E5^%^8C^%^97^%^E4^%^BA^%^AC,BJP&date=2025-07-03&flag=N,N,Y",
        "Sec-Fetch-Dest": "empty",https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc&fs=^%^E4^%
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36 SE 2.X MetaSr 1.0",
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua": "^\\^Not)A;Brand^^;v=^\\^24^^, ^\\^Chromium^^;v=^\\^116^^",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "^\\^Windows^^"
    }
    cookies = {
        "_uab_collina": "175144477944438320557093",
        "JSESSIONID": "3D658D5D08A9CA7B82498DD10544A58C",
        "route": "9036359bb8a8a461c164a04f8f50b252",
        "BIGipServerotn": "1473839370.24610.0000",
        "BIGipServerpassport": "1005060362.50215.0000",
        "guidesStatus": "off",
        "highContrastMode": "defaltMode",
        "cursorStatus": "off",
        "_jc_save_fromStation": "^%^u4E0A^%^u6D77^%^2CSHH",
        "_jc_save_toStation": "^%^u5317^%^u4EAC^%^2CBJP",
        "_jc_save_fromDate": "2025-07-03",
        "_jc_save_toDate": "2025-07-02",
        "_jc_save_wfdc_flag": "dc"
    }

    session = requests.session()
    res = session.get(url, headers=headers, cookies=cookies)

    # ct = res.headers.get("Content-Type","")
    # if "application/json" not in ct:
    #     print("Non-JSON response:", ct)
    #     print(res.text[:2000])   # inspect HTML
    # else:
    #     data = res.json()

    # return
    # data = res.json()
    # print('12306接口返回，并准备后续处理:', data)

    # print("OK????")

    # 这是一个列表
    result = data["data"]["result"]

    lis = []
    for index in result:
        index_list = index.replace('有', 'Yes').replace('无', 'No').split('|')
        # print(index_list)
        train_number = index_list[3]  # 车次

        if 'G' in train_number:
            time_1 = index_list[8]  # 出发时间
            time_2 = index_list[9]  # 到达时间
            prince_seat = index_list[25]  # 特等座
            first_class_seat = index_list[31]  # 一等座
            second_class = index_list[30]  # 二等座
            dit = {
                '车次': train_number,
                '出发时间': time_1,
                '到站时间': time_2,
                "是否可以预定": index_list[11],

            }
            lis.append(dit)
        else:
            # print(index_list)
            time_1 = index_list[8]  # 出发时间
            time_2 = index_list[9]  # 到达时间

            dit = {
                '车次': train_number,
                '出发时间': time_1,
                '到站时间': time_2,
                "是否可以预定": index_list[11],

            }
            lis.append(dit)
    # print(lis)
    content = pd.DataFrame(lis)
    # print(content)
    return content


# 第二个工具，查询当前时间
def check_date():
    today = datetime.now().date()
    return today
