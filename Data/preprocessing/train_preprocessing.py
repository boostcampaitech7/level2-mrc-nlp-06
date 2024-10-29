import pandas as pd
from datasets import load_from_disk
import re
import random
from datasets import Dataset, DatasetDict
import json
import ast

# pandas datasframe으로 변환 및 필요한 컬럼만 추출하는 함수
def dataset_to_df(dataset):
  data_list = []

  for data in dataset:
    data_list.append({
        'title' : data["title"],
        "context" : data["context"],
        "question" : data["question"],
        "id" : data["id"],
        "answer_start" : data["answers"]["answer_start"][0],
        "answer_text" : data["answers"]["text"][0],
        "document_id" : data["document_id"]
    })
  df = pd.DataFrame(data_list)
  return df

# 괄호 안에 있는 외래어 제거 함수
def remove_other_lang(text):
  return re.sub(r"\([^가-힣0-9]*?\)", "", text)

# 생성한 answer_word_idx에서 answer_start와 같은 값을 가지는 인덱스 번호 저장
def get_answer_idx(answer_start, answer_word_idx):
  re_idx = 0
  for idx, data in enumerate(answer_word_idx):
    if data == answer_start:
      re_idx = int(idx)

  return re_idx

# 바뀐 인덱스 값 리턴해주는 함수
def get_after_answer_start(index_list, idx):
  if len(index_list) > idx:
    return index_list[idx]
  return 0

# answer_text와 같은 단어들의 시작 인덱스를 구하는 코드
def get_answer_word_idx(context, answer):
  indices = []
  index = context.find(answer)

  while index != -1:
    indices.append(index)
    index = context.find(answer, index + 1)

  return indices

# answer_start 값이 제대로 변환이 됐는지 확인하는 함수
def checking_good(df):
  t = 0
  f = 0
  total = len(df)
  fail = []

  for i in range(total):
    answer_start = df.loc[i]["answer_start"]
    answer_text = df.loc[i]['answer_text'][0]
    context = df.loc[i]['context'][answer_start]
    if answer_text == context:
      t += 1
    else:
      f += 1
      fail.append(i)
  print(f"answert_start의 값이 총 {total}개 중에서 {t}개가 맞고, {f}개가 틀렸습니다.")
  print(f"성공률 : {(t / total) * 100 : .2f}%")
  if len(fail) > 0:
    print(f"틀린 인덱스 : {fail}")
    return fail
  return []

# 마크 다운 전처리하는 함수
def remove_markdwon(text):
  # 헤더 제거
  text = re.sub(r'^#{1,6}\s', '', text, flags=re.MULTILINE)
  # 볼드체 제거
  text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
  # 이탤릭체 제거
  text = re.sub(r'\*(.*?)\*', r'\1', text)
  # 코드 블록 제거
  text = re.sub(r'```[\s\S]*?```', '', text)
  # 인라인 코드 제거
  text = re.sub(r'`([^`\n]+)`', r'\1', text)
  # 링크 제거 (텍스트만 유지)
  text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
  # 리스트 기호 제거
  text = re.sub(r'^\s*[-*+]\s', '', text, flags=re.MULTILINE)
  return text.strip()

# 마크다운 개수 구하는 함수
def check_markdown(df, column="text"):
  cnt = 0
  text = df[column]
  for text in df[column]:
    # 헤더 개수
    header_count = len(re.findall(r'^#{1,6}\s', text, flags=re.MULTILINE))
    # 볼드체 개수
    bold_count = len(re.findall(r'\*\*(.*?)\*\*', text))
    # 이탤릭체 개수
    italic_count = len(re.findall(r'\*(.*?)\*', text)) - bold_count  # 볼드체가 이탤릭체와 겹침
    # 코드 블록 개수
    code_block_count = len(re.findall(r'```[\s\S]*?```', text))
    # 인라인 코드 개수
    inline_code_count = len(re.findall(r'`([^`\n]+)`', text))
    # 링크 개수
    link_count = len(re.findall(r'\[(.*?)\]\(.*?\)', text))
    # 리스트 기호 개수
    list_item_count = len(re.findall(r'^\s*[-*+]\s', text, flags=re.MULTILINE))

    cnt += (
      header_count +
      bold_count +
      italic_count +
      code_block_count +
      inline_code_count +
      link_count +
      list_item_count
    )

  return cnt

# 괄호 안 외래어 개수 구하는 함수
def check_lang(df, column="text"):
  cnt = 0
  for text in df[column]:
    cnt += len(re.findall(r'\([^가-힣0-9]*?\)', text))
  return cnt

# 개행 제거하는 함수
def clean_text(text):
  # 개행 문자 제거
  text = re.sub(r'\n|\\n', ' ', text)
  # 연속된 공백 제거
  text = re.sub(r'\s+', ' ', text)
  return text.strip()

# 개행 개수 구하는 함수
def check_newline(df, column="text"):
  cnt = 0
  for text in df[column]:
    cnt += len(re.findall(r'\n|\\n', text))
    #cnt = df[column].str.contains("\\n").sum()
  return cnt

# answer 합치는 함수
def combine_answer(df):
  return df[['answer_start', 'answer_text']].apply(lambda x: {'answer_start': [(x['answer_start'])], 'text': [x['answer_text']]}, axis=1)

# 기본 wikipedia 데이터셋 불러오기
def get_wiki(path):
  with open(path, "r") as f:
    wiki = json.load(f)

  df = pd.DataFrame(wiki)
  df = df.transpose()
  return df

# wikipedia 데이터 전처리 main 함수
def wiki_main(df, newline=False, mark=False, lang=False):
  # 작업 전 성능 테스트"
  print("wikipedia Before", "-" * 40)

  # context에 외래어 제거해서 다시 context에 저장 -- 이 위치에 다른 전처리 작업 수행 가능
  if newline:
    print(f"\\n 개수 : {check_newline(df)}")
    df["text"] = df["text"].apply(clean_text)
  if mark:
    print(f"마크다운 총 개수 : {check_markdown(df)}")
    df["text"] = df["text"].apply(remove_markdwon)
  if lang:
    print(f"괄호 안 외래어 총 개수 : {check_lang(df)}")
    df["text"] = df["text"].apply(remove_other_lang)

  # 작업 후 성능 테스트
  print("\nwikipedia After", "-" * 40)
  if newline:
    print(f"\\n 개수 : {check_newline(df)}")
  if mark:
    print(f"마크다운 총 개수 : {check_markdown(df)}")
  if lang:
    print(f"괄호 안 외래어 총 개수 : {check_lang(df)}")
  return df

# 데이터 전처리 main 함수
def preprocessing_main(df, name, newline=False, mark=False, lang=False):
  # 작업 전 성능 테스트
  print(f"{name} Before" , "-" * 40)
  checking_good(df)

  # answer_word_idx라는 컬럼에 정답과 같은 단어들이 시작하는 인덱스 값을 저장
  df["answer_word_idx"] = df.apply(lambda row: get_answer_word_idx(row["context"], row["answer_text"]), axis=1)

  # answer_word_true_idx라는 컬럼에 answer_word_idx를 기준으로 정답 value의 인덱스를 저장
  df["answer_word_true_idx"] = df.apply(lambda row: get_answer_idx(row["answer_start"], row["answer_word_idx"]), axis=1)

  # context에 외래어 제거해서 다시 context에 저장 -- 이 위치에 다른 전처리 작업 수행 가능
  if newline:
    print(f"\\n 개수 : {check_newline(df, 'context')}")
    df["context"] = df["context"].apply(clean_text)
  if mark:
    print(f"마크다운 총 개수 : {check_markdown(df, 'context')}")
    df["context"] = df["context"].apply(remove_markdwon)
  if lang:
    print(f"괄호 안 외래어 총 개수 : {check_lang(df, 'context')}")
    df["context"] = df["context"].apply(remove_other_lang)
    df["answer_text"] = df["answer_text"].apply(remove_other_lang)

  # 전처리 후에 바뀐 context에서 정답과 같은 단어들이 시작하는 인덱스 값을 저장
  df["answer_word_idx_after"] = df.apply(lambda row: get_answer_word_idx(row["context"], row["answer_text"]), axis=1)

  # answer_word_idx_after에서 answer_word_true_idx번재의 인덱스 값을 answer_start에 덮어서 저장
  df["answer_start"] = df.apply(lambda row: get_after_answer_start(row["answer_word_idx_after"], row["answer_word_true_idx"]), axis=1)

  # 작업 후 성능 테스트
  print(f"\n{name} After", "-" * 40)

  fails = checking_good(df)
  if len(fails) > 0:
    print("\n잘 안된 인덱스")
    for fail in fails:
      print(df.loc[fail]["answer_text"])
      print(df.loc[fail]["answer_start"])
      print(df.loc[fail]["context"])
      print(df.loc[fail]["question"])
      print("===============")

  if newline:
    print(f"\\n 개수 : {check_newline(df, 'context')}")
  if mark:
    print(f"마크다운 총 개수 : {check_markdown(df, 'context')}")
  if lang:
    print(f"괄호 안 외래어 총 개수 : {check_lang(df, 'context')}")

  print()
  # 필요 없는 컬럼 삭제
  df = df.drop(columns=["answer_word_idx", "answer_word_true_idx", "answer_word_idx_after"])

  return df

def main():
    # dataset이 저장될 datasets 폴더 경로 지정
    path = "datasets_path_you_want_to_save"

    # 베이스 train_dataset 경로 지정
    train_path = "your_original_train_path"

    # 베이스 test_dataset 경로 지정
    test_path = "your_original_test_dataset_path"
    # 베이스 test_dataset 불러오기
    test_dataset = load_from_disk(test_path)

    # 베이스 wikipedia_document 경로 지정
    wiki_path = "your_original_wikipedia_documents_path"

    # 버전별 파라미터 지정
    # 개행, 마크다운, 외래어 순
    version_para = [[False, False, False],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, True, False],
                    [True, False, True],
                    [False, True, True],
                    [True, True, True]]

    # wikipedia_documents 데이터에서 필요없는 컬럼 선언
    cut = ['corpus_source', 'url', 'domain', 'author', 'html']

    # 8개 버전 돌리기
    for i in range(len(version_para)):

      # 버전별 저장 경로 지정
      save_path = path + "v0.0." + str(i+1)

      # 베이스 train_dataset 불러오기
      train_dataset = load_from_disk(train_path)

      # 데이터 프레임을 바로 Dataset으로 변환
      train_df = dataset_to_df(train_dataset["train"])
      valid_df = dataset_to_df(train_dataset["validation"])

      # train, validation 데이터 프레임 전처리 (version_para 기준으로)
      train_df = preprocessing_main(train_df, "train", newline=version_para[i][0], mark=version_para[i][1], lang=version_para[i][2])
      valid_df = preprocessing_main(valid_df, "valid", newline=version_para[i][0], mark=version_para[i][1], lang=version_para[i][2])

      # 1은 raw 데이터로 변경이 없음. answer_start나 answer_text 컬럼이 없음
      if i != 1:
        train_df['answers'] = combine_answer(train_df)
        train_df = train_df.drop(columns=['answer_start', 'answer_text'])
        valid_df['answers'] = combine_answer(valid_df)
        valid_df = valid_df.drop(columns=['answer_start', 'answer_text'])

      # pandas 데이터 프레임을 다시 dataset으로 변환
      dataset_train = Dataset.from_pandas(train_df)
      dataset_valid = Dataset.from_pandas(valid_df)

      # 저장을 위한 DatasetDict 생성
      dataset_dict = DatasetDict({
          'train': dataset_train,
          'validation': dataset_valid,
          'test': test_dataset["validation"]
      })

      # save_path에 dataset_dict 저장
      dataset_dict.save_to_disk(save_path)

      # 베이스 wikipedia 파일 json 형태에서 DataFrame으로 변환해서 가져오기
      df = get_wiki(wiki_path)

      # wikipedia 데이터 프레임 전처리 (version_para 기준으로)
      df = wiki_main(df, newline=version_para[i][0], mark=version_para[i][1], lang=version_para[i][2])

      # 필요없는 컬럼 삭제
      df = df.drop(cut, axis=1)

      # 데이터 프레임을 다시 json 파일로 저장하기
      df.to_json(save_path + "/wikipedia_documents.json", orient='index', force_ascii=False)

      #
      print()
      print("="*40)
      print(f"v0.0.{i+1} 버전이 저장 되었습니다.")
      print("="*40, "\n")

if __name__ == "__main__":
  main()