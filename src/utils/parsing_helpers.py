import sys
sys.path.append("../../utils")
from definitions import *


def make_delay():
    time.sleep(0.5)


def make_request_and_save_result(url, path):
    subprocess.run(["wget", "-q", "-O", path, url])


def get_pages(year):
    url = f"https://diploma.spbu.ru/gp/index?GpSearch%5Bname_ru%5D=&GpSearch%5Btitle_ru%5D=&GpSearch%5Beditor_ru%5D=&GpSearch%5Bdp_id%5D=&GpSearch%5Bstatus%5D=1&GpSearch%5Byear%5D={year}"
    path = ARTIFACTS_DIR_PATH.joinpath("junk/search_pages.html")
    make_request_and_save_result(url, path)
    with open(path, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    dict_soup = convert(soup)
    return int(dict_soup["html"][0]["body"][0]["div"][0]["div"][1]["div"][0]["div"][0]["div"][1]["div"][0]["div"][1]["div"][0]["b"][1]["#text"].replace(u'\xa0', ''))


def parse_spbu_with_year(year):
    pages = get_pages(year)

    print("Requesting search_pages...")
    base_search_url = "https://diploma.spbu.ru/gp/index?GpSearch%5Bname_ru%5D=&GpSearch%5Btitle_ru%5D=&GpSearch%5Beditor_ru%5D=&GpSearch%5Bdp_id%5D=&GpSearch%5Bstatus%5D=1&GpSearch%5Byear%5D={year}&page={page}"
    base_artifacts_path = ARTIFACTS_DIR_PATH.joinpath(f"parsing/diplomas/spbu/{year}/")
    base_artifacts_path.mkdir(exist_ok=True, parents=True)
    for page in trange(1, pages + 1, desc="Search pages..."):
        make_delay()
        url = base_search_url.format(year=year, page=page)
        search_pages_dir = base_artifacts_path.joinpath("search_pages/")
        search_pages_dir.mkdir(exist_ok=True, parents=True)
        path = search_pages_dir.joinpath(f"{page}.html")
        make_request_and_save_result(url, path)

    print("Collecting ids from search_pages...")
    ids = []
    for page in trange(1, pages + 1, desc="Search pages..."):
        path = base_artifacts_path.joinpath(f"search_pages/{page}.html")
        with open(path, "r") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for x in soup.find_all("a", href=True):
            if "id=" in x["href"]:
                ids.append(x["href"].split('=')[-1])
    with open(base_artifacts_path.joinpath("ids.json"), "w") as f:
        json.dump(ids, f)

    print("Requesting preview by ids...")
    base_view_id_url = "https://diploma.spbu.ru/gp/view?id={}"
    dir_path = base_artifacts_path.joinpath("preview/")
    dir_path.mkdir(exist_ok=True, parents=True)
    for id in tqdm(ids, desc="Ids..."):
        make_delay()
        url = base_view_id_url.format(id)
        path = dir_path.joinpath(f"{id}.html")
        make_request_and_save_result(url, path)

    print("Requesting view by preview...")
    base_view_work_url = "https://dspace.spbu.ru/handle"
    new_dir_path = base_artifacts_path.joinpath("view/")
    new_dir_path.mkdir(exist_ok=True, parents=True)
    for id in tqdm(ids, desc="Ids..."):
        make_delay()
        path = base_artifacts_path.joinpath(f"preview/{id}.html")
        with open(path, "r") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        new_path = f"{new_dir_path}/{id}.html"
        for x in soup.find_all("a", href=True):
            if "hdl" in x["href"]:
                id1 = x["href"].split('/')[-2]
                id2 = x["href"].split('/')[-1]
                new_url = f"{base_view_work_url}/{id1}/{id2}"
                make_request_and_save_result(new_url, new_path)

    print("Requesting doc by view...")
    base_work_doc_url = "https://dspace.spbu.ru/{}"
    new_dir_path = base_artifacts_path.joinpath("work/")
    new_dir_path.mkdir(exist_ok=True, parents=True)
    skipped_ids = []
    for id in tqdm(ids, desc="Ids..."):
        make_delay()
        path = base_artifacts_path.joinpath(f"view/{id}.html")
        try:
            with open(path, "r") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
        except Exception:
            skipped_ids.append({
                "id": id,
                "reason": "no work",
            })
            continue
        soup_dict = convert(soup)
        summary = None
        table_meta = soup_dict['html'][0]['head'][0]['meta'][3]['meta']
        for row in table_meta:
            if row["@name"] == "DCTERMS.abstract" and row["@xml:lang"] == "ru_RU":
                summary = row["@content"]
                break
        if summary is None:
            skipped_ids.append({
                "id": id,
                "reason": "summary",
            })
            continue
        divs = soup_dict['html'][0]['body'][0]['main'][0]['div'][2]['div']
        bitstream = None
        for div in divs:
            if div.get('table', [{}])[0].get('tr', [{}, {}])[1].get('td', [{}])[0].get('a', [{}])[0].get("@href"):
                bitstream = div['table'][0]['tr'][1]['td'][0]['a'][0]["@href"]
                break
        if bitstream is None:
            skipped_ids.append({
                "id": id,
                "reason": "bitstream",
            })
            continue

        with open(new_dir_path.joinpath(f"{id}_abstract.txt"), "w") as f:
            f.write(summary)
            
        content_url = base_work_doc_url.format(bitstream)
        ext = content_url.split(".")[-1]
        make_request_and_save_result(content_url, new_dir_path.joinpath(f"{id}_diploma.{ext}"))
    print(f"{len(skipped_ids)} ids was skipped")
    
    with open(base_artifacts_path.joinpath("skipped_ids.json"), "w") as f:
        json.dump(skipped_ids, f)